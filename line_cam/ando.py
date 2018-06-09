import cv2
import numpy as np
#import pytesseract as pt

class Camera_Line():
    flag = -1
    #-1 = not detected
    #0 = go throught
    #1 = go right
    # = go left
    
    roiPx1 = 280
    roiPx2 = 480
    roiPy1 = 1
    roiPy2 = 639
    
    def radianToDegree(self,angle):
        return theta / np.pi * 180
    #########################radian to degree

    def getLines(self,src, lowthreshold=50, highthreshold=300, mode=1):
        src = src[self.roiPx1:self.roiPx2, self.roiPy1:self.roiPy2] 
        #src = src[]
        Edges = cv2.Canny(src, lowthreshold, highthreshold, None, 3)
        #cv2.imshow("CannyEditedPicture", Edges)
        Lines = cv2.HoughLines(Edges, 1, np.pi/180, 90, None, 0,0)
        if Lines is not None:
            for indexh in range(len(Lines)):
                for rh, thetah in Lines[indexh]:
                    ah = np.cos(thetah)
                    bh = np.sin(thetah)
                    x0h = ah*rh
                    y0h = bh*rh
                    x1h = int(x0h + 1000*(-bh))
                    x2h = int(x0h - 1000*(-bh))
                    y1h = int(y0h + 1000*ah)
                    y2h = int(y0h - 1000*ah)

                    if mode is 1:
                        cv2.line(src, (x1h,y1h), (x2h,y2h), (0,0,255), 2)
                    #print(thetah)

                    #rint(theta / np.pi * 180)
                    #cv2.imshow("LineEditedPicture", src)
                    
            return Lines
        else:
            pass
    ########################################## get line to hough

    def IntersectPoint(self,x1,x2,x3,x4,y1,y2,y3,y4):
        under = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if(under == 0):
            #print("trash2")
            return -1,0,0
        
        tt = (x4-x3) * (y1-y3) - (y4-y3)*(x1-x3)
        ss = (x2-x1) * (y1-y3) - (y2-y1)*(x1-x3)
        
        t = tt/under
        s = ss/under
        
        if(t < 0 or t== 1 or s<0 or s==1):
            #print("trash1")
            return -1,0,0
        if(tt==0 and ss==0):
            #print("trash3")
            return -1,0,0
        
        x = x1 + t * (x2-x1)
        y = y1 + t * (y2-y1)
        #print("Pass")
        return 1,x,y
    ##########################################get Cross point in fnc

    def getCrossPoint(self,img,lines,mode = 1) :
        PointX = []
        PointY = []
        pointindex = 0
        for indexp1 in range(len(lines)):
            for rp1, thetap1 in lines[indexp1]:
                ap1 = np.cos(thetap1)
                bp1 = np.sin(thetap1)
                x0p1 = ap1*rp1
                y0p1 = bp1*rp1
                x1p1 = int(x0p1 + 50*(-bp1))
                x2p1 = int(x0p1 - 50*(-bp1))
                y1p1 = int(y0p1 + 50*ap1)
                y2p1 = int(y0p1 - 50*ap1)
                
                for index2 in range(len(lines)):
                    for r2, theta2 in lines[index2]:
                        a2 = np.cos(theta2)
                        b2 = np.sin(theta2)
                        x00 = a2*r2
                        y00 = b2*r2
                        x3 = int(x00 + 50*(-b2))
                        x4 = int(x00 - 50*(-b2))
                        y3 = int(y00 + 50*a2)
                        y4 = int(y00 - 50*a2)
                        if((thetap1-theta2) < 0.2 and (thetap1-theta2) > -0.2):
                            continue
                        
                        bool,X,Y = self.IntersectPoint(x1p1+self.roiPy1,x2p1+self.roiPy1,x3+self.roiPy1,x4+self.roiPy1,y1p1+self.roiPx1,y2p1+self.roiPx1,y3+self.roiPx1,y4+self.roiPx1)
                        
                        if(bool < 0 or Y>320 or X<150 or X>490 ):
                            pass
                        else:
                            #print(X+self.roiPy1,Y+self.roiPx1)
                            #img = cv2.circle(img,(int(X+self.roiPy1),int(Y+self.roiPx1)),5,(255,255,255),-1)
                            PointX.append(X)
                            PointY.append(Y)
        return PointX,PointY

    ########################################## put crossed point in lines
            
    def AvgPoint(self,PointX,PointY,img):
        Xsum = 0
        Ysum = 0
        X=0
        Y=0
        for I in range(len(PointX)):
            Xsum+=PointX[I]
        for I in range(len(PointY)):
            Ysum+=PointY[I]
        if(len(PointX)==0):
            return -1,-1
        else:
            X=int(Xsum/len(PointX))
        if(len(PointY)==0):
            return -1,-1
        else:
            Y=int(Ysum/len(PointY))
        #img = cv2.circle(img,(X,Y),5,(255,0,255),-1)
        
        return X,Y
    ################################averange crossed point
 
    def set_flag(self,x,y):
        if(x==-1 or y == -1):
            return -1
        elif(x>=280 and x<=360):
            return 0
        elif(x<280):
            return 1
        elif(x>340):
            return 2
        else:
            return -1
            
    ############################## set the flag type
 
    def DrawSizeline(self,img):
        indexXT = 0
        indexYT = 0
        while(True):
            indexYT = 0
            while(True):
                img = cv2.circle(img,(indexXT,indexYT),8,(0,0,0),-1)
                indexYT += 100;
                if(indexYT > 1000):
                    break
            if(indexXT > 1000):
                break
            indexXT += 100;
        cv2.line(img, (320,0), (320,1000), (0,255,255), 2)
        print(img.shape)
    ############################### draw vertical lines

    def main(self):
        indextemp = 0 #for counting index or else
        Cam = cv2.VideoCapture(0)
        return_value, img_Cam = Cam.read()

        lines = []
        points = []
        pointX = []
        pointY = []
        CrossedX=-1
        CrossedY=-1

        while True:
            pointX[0:] = []
            pointY[0:] = []
            return_value, img_Cam = Cam.read()

            img_Cam = cv2.flip(img_Cam, 0)
            img_Cam = cv2.flip(img_Cam, 1)
            #####flip the cam

            lines = self.getLines(img_Cam, 200, 300, 1)
            if lines is not None:
                pointX, pointY = self.getCrossPoint(img_Cam,lines,1)
                if pointX is not None and pointY is not None:
                    CrossedX,CrossedY = self.AvgPoint(pointX,pointY,img_Cam)
                    
            #self.DrawSizeline(img_Cam)
            #cv2.rectangle(img_Cam,(self.roiPy1,self.roiPx1),(self.roiPy2,self.roiPx2),(255,255,0),2)    #draw roi
            #cv2.imshow("LineEditedPicture", img_Cam)
            self.flag = self.set_flag(CrossedX,CrossedY)
            print(self.flag)
            if cv2.waitKey(10) == 27:
                break  # esc to quit
            
            CrossedX=-1;
            CrossedY=-1;
        cv2.destroyAllWindows()
    
    ###############################################    main

if __name__ == "__main__":
    m = Camera_Line()
    m.main()