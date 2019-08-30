import argparse
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tarfile

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
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = Image.fromarray(gray)
    _,seg_map = model.run(gray)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    bin_seg_image = (seg_image > 127)*255
    return cv2.bitwise_and(frame,frame,mask=bin_seg_image)

#Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Backgroud Removal")
    parser.add_argument("trimap",type=str,nargs='?',default='',help="Path to the trimap model")
    return parser.parse_args()

def main(args):

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("EPN-Photo")
    background = cv2.imread("background.png")
    SEGMENTATION_MODEL_PATH = 'segmentation_models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
    MODEL = DeepLabModel(SEGMENTATION_MODEL_PATH)
    img_counter = 0

    while True:
        ret, frame = cam.read()
        
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
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            #img_nobg = removeBackground(frame,background)
            img_nobg = removeBackground2(frame,MODEL)
            img_name_nobg = "opencv_frame_nobg_{}.png".format(img_counter)
            cv2.imwrite(img_name_nobg, img_nobg)
            img_counter += 1

        cv2.imshow("EPN-Photo", frame)

    cam.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Add deeplab to pythonpath
    args = parse_args()
    main(args)


