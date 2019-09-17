import argparse
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tarfile
from datetime import datetime, timedelta
import time
from InstagramAPI import InstagramAPI
import configparser
from playsound import playsound
import os, random

from helper import DeepLabModel

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

def removeBackground(frame,model):
    # Convert it to grayscale
    height,width,_ = frame.shape
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

def setCurrentBackground(bg_index,width,length,bg_path,current_bgs):
    if len(current_bgs) == 0:
        current_bgs = os.listdir(bg_path)
    random.seed(random.randrange(10,1000))
    newbg = cv2.imread(bg_path+'/'+current_bgs.pop(random.randrange(len(current_bgs))))
    newbg = cv2.resize(newbg,(width,length))
    return newbg,current_bgs

def takePicture(img_counter,img):
    playsound('sounds/Camera_Click.mp3')
    img_name = "opencv_frame_nobg_{}.jpg".format(img_counter)
    print("{} written!".format(img_name))
    cv2.imwrite(img_name, img)
    return img_name

def uploadImageInstagram(img_name,user,passw,caption="Testing"):
    api = InstagramAPI(user, passw)
    if (api.login()):
        api.getSelfUserFeed()  # get self user feed
        print(api.LastJson)  # print last response JSON
        print("Login succes!")
        #api.uploadPhoto(img_name, caption=caption)
    else:
        print("Can't login!")

#Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Backgroud Removal")
    parser.add_argument("trimap",type=str,nargs='?',default='',help="Path to the trimap model")
    conf = configparser.ConfigParser()
    conf.read('main.config')
    return parser.parse_args(),conf

def main(args,conf):

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("EPN-Photo")
    #cv2.setWindowProperty('EPN-Photo', cv2.WND_PROP_FULLSCREEN, 1)
    SEGMENTATION_MODEL_PATH = conf.get('SEGMENTATION','SegmentationModelPath')
    MODEL = DeepLabModel(SEGMENTATION_MODEL_PATH)
    BACKGROUND_PATH = conf.get('BACKGROUND','backgroundPath')
    curr_bgs = os.listdir(BACKGROUND_PATH)
    bg_counter = 0
    img_counter = 0
    WIDTH = int(cam.get(3))
    HEIGHT = int(cam.get(4))
    period = timedelta(seconds=int(conf.get('TIMER','TimerPeriod')))
    newbg,curr_bgs = setCurrentBackground(bg_counter,int(cam.get(3)),int(cam.get(4)),BACKGROUND_PATH,curr_bgs)
    picture_flag = False
    background_flag = False

    while True:
        ret, frame = cam.read()

        if background_flag:
            img_nobg,seg_image = removeBackground(frame,MODEL)
            img_nobg = addBackground(img_nobg,newbg,seg_image)
            img_nobg_2 = img_nobg
        else:
            img_nobg = frame
        
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 98:
            
            if background_flag:
                background_flag = False
                print("Removing background...")
            else:
                background_flag = True
                print("Adding background...")
        elif k%256 == 32:
            # SPACE pressed
            next_time = datetime.now() + period
            picture_flag = True
        elif k%256 == 3:
            # RIGHT ARROW pressed - Get previous background
            playsound('sounds/Robot_blip.mp3')
            bg_counter += 1
            newbg,curr_bgs = setCurrentBackground(bg_counter,WIDTH,HEIGHT,BACKGROUND_PATH,curr_bgs)
        elif k%256 == 2:
            # LEFT ARROW pressed - Get next background
            playsound('sounds/Robot_blip.mp3')
            bg_counter -= 1
            if bg_counter < 0:
                bg_counter = 0
            newbg,curr_bgs = setCurrentBackground(bg_counter,WIDTH,HEIGHT,BACKGROUND_PATH,curr_bgs)
        elif picture_flag == True:
            n = datetime.now()
            if next_time >= n:
                playsound('sounds/Tick.mp3')
                diff = (next_time-n).total_seconds()
                cv2.putText(img_nobg,str(int(diff)),(WIDTH//2-100,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,4.0,(255,255,255),10)
            else:   
                #cv2.putText(img_nobg,"SONRIE",(WIDTH//2-100,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,4.0,(255,255,255),10)
                img_name = takePicture(img_counter,img_nobg)
                img_counter += 1
                picture_flag = False
                time.sleep(3)
                cv2.imshow("EPN-Photo", img_nobg_2)
                # INSTAGRAM UPLOAD
                if eval(conf.get('INSTAGRAM','sendInstagram'))==True:
                    cv2.putText(img_nobg,"SUBIENDO INSTAGRAM",(WIDTH//2-200,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,2.0,(255,255,255),10)
                    uploadImageInstagram(img_name,conf.get('INSTAGRAM','username'),conf.get('INSTAGRAM','password'),conf.get('INSTAGRAM','caption'))
                    time.sleep(3)

        cv2.imshow("EPN-Photo", img_nobg)

    cam.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Add deeplab to pythonpath
    args,conf = parse_args()
    main(args,conf)


