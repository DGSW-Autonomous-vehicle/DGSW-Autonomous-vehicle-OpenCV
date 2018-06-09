from PIL import Image
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    cv2.imshow("LUL", img)

    key = cv2.waitKey(1)
    if key == 27:
        break
glPixelZoom(1, -1)
cv2.destroyAllWindows()

