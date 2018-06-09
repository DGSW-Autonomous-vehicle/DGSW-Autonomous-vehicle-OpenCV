import cv2

xhost = +local:root
xhost = +localhost
cap = cv2.VideoCapture(0) #0 or -1
while cap.isOpened():
	ret, img = cap.read()
	if ret:
		cv2.imshow('camera-0', img)
	if cv2.waitKey(1) & 0xFF == 27:
		break
	else:
		print('no camera!')
		break
cap.release()
cv2.destroyAllWindows()
