import argparse
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tarfile
from datetime import datetime, timedelta
import time
from InstagramAPI import InstagramAPI

from helper import DeepLabModel

BG_LIST = ['P1013972','P1014053','P1014083','P1014335','P1014503','P1014572','P1014608','P1014644','P1014653','P1014654']

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def removeBackground(frame,background):
    # Convert it to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
    fg = cv2.absdiff(gray,gray_bg)
    # Threshold it
    th,threshed = cv2.threshold(fg,10,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # Find min area contour
    _cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(_cnts, key=cv2.contourArea)
    # for cnt in cnts:
    #     if cv2.contourArea(cnt)>1e15:
    #         break
    # Create mask bitwise op
    mask = np.zeros(frame.shape[:2],np.uint8)
    cv2.drawContours(mask,cnts,-1,255,-1)
    dst = cv2.bitwise_and(gray,gray,mask=mask)
    return dst

def removeBackground2(frame,model):
    # Convert it to grayscale
    height,width,channels = frame.shape
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = Image.fromarray(gray)
    _,seg_map = model.run(gray)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image = cv2.cvtColor(seg_image,cv2.COLOR_BGR2GRAY)
    seg_image = cv2.resize(seg_image,(width,height))
    out = cv2.bitwise_and(frame,frame,mask=seg_image)
    return out,seg_image

def addBackground(img,bg,seg_image):
    newbg = cv2.bitwise_and(bg,bg,mask=np.max(seg_image)-seg_image)
    out = cv2.addWeighted(img,1,newbg,0.9,0)
    return out

def setCurrentBackground(bg_index,width,length):
    idx = bg_index % len(BG_LIST)
    newbg = cv2.imread('backgrounds/'+str(BG_LIST[idx])+'.JPG')
    newbg = cv2.resize(newbg,(width,length))
    return newbg

def takePicture(img_counter,img):
    img_name = "opencv_frame_nobg_{}.jpg".format(img_counter)
    print("{} written!".format(img_name))
    cv2.imwrite(img_name, img)
    return img_name

def uploadImageInstagram(img_name):
    api = InstagramAPI("user", "pass")
    if (api.login()):
        api.getSelfUserFeed()  # get self user feed
        print(api.LastJson)  # print last response JSON
        print("Login succes!")
        caption = "Testing"
        api.uploadPhoto(img_name, caption=caption)
    else:
        print("Can't login!")

#Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Backgroud Removal")
    parser.add_argument("trimap",type=str,nargs='?',default='',help="Path to the trimap model")
    return parser.parse_args()

def main(args):

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("EPN-Photo")
    SEGMENTATION_MODEL_PATH = 'segmentation_models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
    MODEL = DeepLabModel(SEGMENTATION_MODEL_PATH)
    bg_counter = 0
    img_counter = 0
    WIDTH = int(cam.get(3))
    HEIGHT = int(cam.get(4))
    period = timedelta(seconds=5)
    newbg = setCurrentBackground(bg_counter,int(cam.get(3)),int(cam.get(4)))
    picture_flag = False

    while True:
        ret, frame = cam.read()

        img_nobg,seg_image = removeBackground2(frame,MODEL)
        img_nobg = addBackground(img_nobg,newbg,seg_image)
        
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            #cv2.putText(frame,"HOLA - TOMANDO FOTO",(100,150),cv2.FONT_HERSHEY_TRIPLEX,0.9,(255,255,255))
            next_time = datetime.now() + period
            picture_flag = True
        elif k%256 == 3:
            # RIGHT ARROW pressed - Get previous background
            bg_counter += 1
            newbg = setCurrentBackground(bg_counter,int(cam.get(3)),int(cam.get(4)))
        elif k%256 == 2:
            # LEFT ARROW pressed - Get next background
            bg_counter -= 1
            if bg_counter < 0:
                bg_counter = 0
            newbg = setCurrentBackground(bg_counter,int(cam.get(3)),int(cam.get(4)))
        elif picture_flag == True:
            n = datetime.now()
            if next_time >= n:
                diff = (next_time-n).total_seconds()
                cv2.putText(img_nobg,str(int(diff)),(WIDTH//2-100,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,4.0,(255,255,255),10)
            else:    
                img_name = takePicture(img_counter,img_nobg)
                img_counter += 1
                picture_flag = False
                cv2.putText(img_nobg,"SONRIE",(WIDTH//2-100,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,4.0,(255,255,255),10)
                cv2.imshow("EPN-Photo", img_nobg)
                time.sleep(3)
                cv2.putText(img_nobg,"SUBIENDO INSTAGRAM",(WIDTH//2-200,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,4.0,(255,255,255),10)
                uploadImageInstagram(img_name)

        cv2.imshow("EPN-Photo", img_nobg)

    cam.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Add deeplab to pythonpath
    args = parse_args()
    main(args)


