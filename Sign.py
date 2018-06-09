import numpy as np
import cv2
from time import sleep


def getSigns():

    signs = []

    sign_left = cv2.imread("./signs/Left.png", cv2.IMREAD_GRAYSCALE)
    sign_slow =     cv2.imread("./signs/Slow.png", cv2.IMREAD_GRAYSCALE)
    sign_stop =     cv2.imread("./signs/Stop.png", cv2.IMREAD_GRAYSCALE)
    sign_speed =    cv2.imread("./signs/speed.png", cv2.IMREAD_GRAYSCALE)

    signs.append(sign_left)
    signs.append(sign_slow)
    signs.append(sign_stop)
    signs.append(sign_speed)

    return signs


def getSignORB(signs):
    orb = cv2.ORB_create()
    kp = []
    des = []

    for i in range(len(signs)):
        kp_, des_ = orb.detectAndCompute(signs[i], None)
        kp.append(kp_)
        des.append(des_)

    return kp, des


def BFMatch(signs, img, kp, des, contours):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB_create()

    copy_img = img.copy()
    copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("check1",signs[0])
    #cv2.imshow("check2", signs[1])
    #cv2.imshow("check3", signs[2])
    #cv2.imshow("check4", signs[3])
    for i in range(len(signs)):
        for i2 in range(len(contours)):

            x = contours[i2][0]
            y = contours[i2][1]
            w = contours[i2][2]
            h = contours[i2][3]

            conImage = copy_img[y:y+h, x:x+w]
            kp_, des_ = orb.detectAndCompute(conImage, None)
            if des_ is None or kp_ is None:
                continue

            matches = bf.match(des[i], des_)

            #matches = sorted(matches, key=lambda x:x.distance)

            img = cv2.drawMatches(signs[i], kp[i], contours[i2], kp_, matches[:50], copy_img, flags=2)
            
            
            
            if len(matches) >= 45 and i == 3:
                cv2.imshow("check1", signs[3])
                #sleep(0.1)
            if len(matches) >= 45 and i == 2:
                cv2.imshow("check2", signs[2])
                #sleep(0.1)
            if len(matches) >= 45 and i == 1:
                cv2.imshow("check3", signs[1])
                #sleep(0.1)
            if len(matches) >= 45 and i == 0:
                cv2.imshow("check4", signs[0])
                #sleep(0.1)





def getContours(image):

    copy_img = image.copy()
    copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(copy_img, 100, 200, None, 3)

    cnt, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def drawContours(image, contours, low_size):
    box = []

    for i in range(len(contours)):
        c = contours[i]
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)

        rect_area = w*h
        ratio = float(w/h) # 기울기

        if(ratio >= 0.2) and (ratio <= 1.0) and (rect_area >= low_size) and (rect_area <= 40000):
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            box.append(cv2.boundingRect(c))
    return image, box


def main():
    Cam = cv2.VideoCapture(0)
    Signs = getSigns()
    kp, des = getSignORB(Signs)

    while True:
        return_value, img_Cam = Cam.read()
        img_Cam = cv2.flip(img_Cam, 0)
        img_Cam = cv2.flip(img_Cam, 1)

        if img_Cam is None:
            print("No camera")
            break

        contours = getContours(img_Cam)

        if contours is not None:
            img_Cam, box = drawContours(img_Cam, contours, 1400)

            BFMatch(Signs, img_Cam, kp, des, box)

        cv2.imshow("a", img_Cam)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()