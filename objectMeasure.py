import cv2
import numpy as np
import utils

webCam=False
path="./images/2022011014150371.jpg"
cap=cv2.VideoCapture(0)
cap.set(10,160) #cv2.CAP_PROP_BRIGHTNESS
cap.set(3,1920)
cap.set(4,1080)


while True:
        if webCam:
          success,img=cap.read()
        else:
                img=cv2.imread(path)
        img,finlaContours=utils.getContours(img,showCanny=True,draw=True)
        
        
        img=cv2.resize(img,(0,0),None,0.5,0.5) 
                
        cv2.imshow("original",img)
        
        if cv2.waitKey(1) & 0xFF==ord("q"):
                cv2.destroyAllWindows()
                break

        
	
                
	        
               
		
		