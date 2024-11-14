import ultralytics
from ultralytics import YOLO
import cvzone
import cv2
import math

model=YOLO('best.pt')

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest',
                'machinery', 'vehicle']

cap=cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 360)

while True:
    success, img=cap.read()
    results=model(img, stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1

            
            conf = (math.ceil(box.conf[0]*100))/100
            cls = int(box.cls[0])
            if((class_names[cls]=="Hardhat" or class_names[cls]=="Mask" or class_names[cls]=="Safety Vest") and (conf>=0.3)):
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0,255,0))
            elif((class_names[cls]=="NO-Hardhat" or class_names[cls]=="NO-Mask" or class_names[cls]=="NO-Safety Vest") and (conf>=0.3)):
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0,0,255))
            else:   
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255,0,255))


            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{conf} {class_names[cls]}",
                                (max(0, x1), max(35, y1)), scale=1, thickness=1)
            
    cv2.imshow("WEBCAM", img)

    if (cv2.waitKey(1) == ord('q')):
        break
        
cap.release()
cv2.destroyAllWindows()