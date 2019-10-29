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
import shelve
import requests

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

def initCurrentBackgrounds(path):
    current_bgs={}
    available_folders = list(filter(lambda x: os.path.isdir(path+'/'+x), os.listdir(path)))
    for f in available_folders:
        current_bgs[f] = []
    return current_bgs

def pickFolder(path):
    available_folders = list(filter(lambda x: os.path.isdir(path+'/'+x), os.listdir(path)))
    return random.choice(available_folders)

def setCurrentBackground(bg_index,width,length,bg_path,current_bgs):
    bg_cat = pickFolder(bg_path)
    bg_path = bg_path+'/'+bg_cat
    if len(current_bgs[bg_cat]) == 0:
        current_bgs[bg_cat] = os.listdir(bg_path)
    random.seed(random.randrange(10,1000))
    newbg = cv2.imread(bg_path+'/'+current_bgs[bg_cat].pop(random.randrange(len(current_bgs[bg_cat]))))
    newbg = cv2.resize(newbg,(width,length))
    return newbg,bg_cat,current_bgs

def takePicture(img_counter,img):
    playsound('sounds/Camera_Click.mp3')
    d = datetime.now()
    img_name = "reprogramar_epn_{}_{}.jpg".format(img_counter,d.strftime("%d_%m_%Y"))
    print("{} written!".format(img_name))
    cv2.imwrite(img_name, img)
    return img_name

def uploadImageInstagram(conf,img_name,caption="Testing"):
    try:
        api = InstagramAPI(conf.get('INSTAGRAM','username'),conf.get('INSTAGRAM','password'))
        if eval(conf.get('INSTAGRAM','sendInstagram'))==True:
            try:
                if (api.login()):
                    api.getSelfUserFeed()  # get self user feed
                    print("Login succes!")
            except requests.exceptions.RequestException as e:
                    print(e)
                    print("Can't login!")

        if (hasattr(api,'token')):
            api.uploadPhoto(img_name, caption=caption)
            print("upload succeed "+img_name)
        else:
            print("upload failed")
            os.rename(img_name,"pending_"+img_name)
    except requests.exceptions.RequestException as e:
        print(e)
        os.rename(img_name,"pending_"+img_name)

#Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Backgroud Removal")
    parser.add_argument("trimap",type=str,nargs='?',default='',help="Path to the trimap model")
    conf = configparser.ConfigParser()
    conf.read('main.config')
    return parser.parse_args(),conf

def main(args,conf,db):

    cam = cv2.VideoCapture(0)
    header = "online"

    cv2.namedWindow("Objeto-Selfie-Humano ("+header+")",cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Objeto-Selfie-Humano ('+header+")", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
    SEGMENTATION_MODEL_PATH = conf.get('SEGMENTATION','SegmentationModelPath')
    MODEL = DeepLabModel(SEGMENTATION_MODEL_PATH)
    
    bg_counter = 0
    img_counter = db['img_counter']
    WIDTH = int(cam.get(3))
    HEIGHT = int(cam.get(4))
    period = timedelta(seconds=int(conf.get('TIMER','TimerPeriod')))

    BACKGROUND_PATH = conf.get('BACKGROUND','backgroundPath')
    curr_bgs = initCurrentBackgrounds(BACKGROUND_PATH)
    newbg,bg_cat,curr_bgs = setCurrentBackground(bg_counter,int(cam.get(3)),int(cam.get(4)),BACKGROUND_PATH,curr_bgs)
    picture_flag = False
    background_flag = True

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)

        if background_flag:
            img_nobg,seg_image = removeBackground(frame,MODEL)
            img_nobg = addBackground(img_nobg,newbg,seg_image)
            img_nobg_2 = img_nobg
            cv2.putText(img_nobg,bg_cat.replace("_"," "),(WIDTH-len(bg_cat)*10-60,HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        else:
            img_nobg = frame
        
        if not ret:
            break
        k = cv2.waitKey(1)
        K = k%256

        if K == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif K == 98:
            # CHANGE BACKGROUND
            if background_flag:
                background_flag = False
                print("Removing background...")
            else:
                background_flag = True
                print("Adding background...")
        elif K == 32 or K == 13:
            # SPACE pressed
            next_time = datetime.now() + period
            picture_flag = True
        elif K == 54 or K == 15:
            # RIGHT ARROW pressed - Get previous background
            playsound('sounds/Robot_blip.mp3')
            bg_counter += 1
            newbg,bg_cat,curr_bgs = setCurrentBackground(bg_counter,WIDTH,HEIGHT,BACKGROUND_PATH,curr_bgs)
        elif K == 52 or K == 14:
            # LEFT ARROW pressed - Get next background
            playsound('sounds/Robot_blip.mp3')
            bg_counter -= 1
            if bg_counter < 0:
                bg_counter = 0
            newbg,bg_cat,curr_bgs = setCurrentBackground(bg_counter,WIDTH,HEIGHT,BACKGROUND_PATH,curr_bgs)
        elif picture_flag == True:
            n = datetime.now()
            if next_time > n:
                playsound('sounds/Tick.mp3')
                diff = (next_time-n).total_seconds()
                if int(diff) == 0:
                    cv2.putText(img_nobg,"SONRIE",(WIDTH//2-100,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,2.0,(255,255,255),4)
                else:
                    cv2.putText(img_nobg,str(int(diff)),(WIDTH//2-100,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,4.0,(255,255,255),10)
            else:   
                print("taking picture")
                img_nobg = img_nobg_2
                #cv2.putText(img_nobg,"SONRIE",(WIDTH//2-100,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,4.0,(255,255,255),10)
                img_name = takePicture(img_counter,img_nobg)
                img_counter += 1
                db['img_counter'] = img_counter
                picture_flag = False
                cv2.imshow("Objeto-Selfie-Humano ("+header+")", img_nobg_2)
                # INSTAGRAM UPLOAD
                if eval(conf.get('INSTAGRAM','sendInstagram'))==True:
                    uploadImageInstagram(conf,img_name,conf.get('INSTAGRAM','caption'))
                    cv2.putText(img_nobg,"INSTAGRAM",(WIDTH//2-180,HEIGHT//2),cv2.FONT_HERSHEY_DUPLEX,2.0,(255,255,255),4)
                    #time.sleep(3)

        cv2.imshow("Objeto-Selfie-Humano ("+header+")", img_nobg)
    cam.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Add deeplab to pythonpath
    db = shelve.open("reprogramar_epn_db")
    if not 'img_counter' in db:
        db['img_counter'] = 0

    args,conf = parse_args()
    main(args,conf,db)


