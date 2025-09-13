import cv2
from ultralytics import YOLO

model=YOLO("helmet.pt")

img=cv2.imread("helmet.jpeg")
result=model(img)
count=0
for r in result:
    for box in r.boxes:
        cls=int(box.cls[0])
        label=model.names[cls]
        conf=float(box.conf[0])

        if conf > 0.5:
            count=count+1
            (x,y,w,h)=map(int,box.xyxy[0])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            cv2.putText(img,f"{label}{count}:{conf:.2f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,)


    print(r)
resize=cv2.resize(img,(1000,600))
cv2.imshow("image",resize)
cv2.waitKey(0)