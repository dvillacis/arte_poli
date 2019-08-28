import cv2
import numpy as np

def removeBackground(frame,background):
    # Convert it to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
    diff = np.abs(gray_bg - gray)
    # Threshold it
    th,threshed = cv2.threshold(diff,127,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # Find min area contour
    _cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(_cnts, key=cv2.contourArea)
    for cnt in cnts:
        if cv2.contourArea(cnt)>10000000:
            break
    # Create mask bitwise op
    mask = np.zeros(frame.shape[:2],np.uint8)
    cv2.drawContours(mask,[cnt],-1,255,-1)
    dst = cv2.bitwise_and(frame,frame,mask=mask)
    return dst

def removeBackground2(frame):
    # Convert it to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Create mask
    
    fgmask = fgbg.apply(gray)
    #dst = cv2.bitwise_and(frame,frame,mask=fgmask)
    return fgmask

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
background = cv2.imread("background.png")

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
        cv2.putText(frame,"HOLA - TOMANDO FOTO",(100,150),cv2.FONT_HERSHEY_TRIPLEX,0.9,(255,255,255))
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_nobg = removeBackground2(frame)
        img_name_nobg = "opencv_frame_nobg_{}.png".format(img_counter)
        cv2.imwrite(img_name_nobg, img_nobg)
        img_counter += 1

    cv2.imshow("test", frame)

cam.release()

cv2.destroyAllWindows()


