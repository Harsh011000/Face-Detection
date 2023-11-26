from ultralytics import YOLO
import cv2
import face_fetch
import face_embeddings_test
import push_data

model=YOLO("face-model.pt")

cv2.setUseOptimized(True)  # Enable optimization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    true,frame=cap.read()
    if true:
        results = model.predict(frame, conf=0.7, iou=0.3,device=0)
        result=results[0]
        face_lis=[]
        for box in result.boxes:
            xywh=face_fetch.extract_face(box.xyxy[0].tolist(),frame)
            name=face_embeddings_test.match()
            tmp=[name,xywh]
            face_lis.append(tmp)
        annote=result.plot()
        for x in face_lis:
            print(x)
            cv2.putText(annote,x[0], (int(x[1][0]), int((x[1][1]-50))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

        cv2.imshow("detection",annote)
        if cv2.waitKey(1)==32:
            chk=push_data.store()
            print(face_embeddings_test.flag)
            face_embeddings_test.flag=0
        if cv2.waitKey(1)==27:
            break;
cap.release()
cv2.destroyAllWindows()