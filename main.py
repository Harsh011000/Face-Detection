from ultralytics import YOLO
import cv2
import face_fetch
import face_embeddings_test
import push_data

model=YOLO("face-model.pt")

cv2.setUseOptimized(True)  # Enable optimization
zoom_scale = 1.0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    true,frame=cap.read()
    if true:
        center_x, center_y = int(frame.shape[1] / 2), int(frame.shape[0] / 2)

        # Calculate the cropping region
        crop_width = int(frame.shape[1] / zoom_scale)
        crop_height = int(frame.shape[0] / zoom_scale)

        # Calculate the crop coordinates
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        x2 = min(frame.shape[1], center_x + crop_width // 2)
        y2 = min(frame.shape[0], center_y + crop_height // 2)

        # Crop and resize the frame
        frame_zoomed = frame[y1:y2, x1:x2]
        frame_zoomed = cv2.resize(frame_zoomed, (frame.shape[1], frame.shape[0]))



        results = model.predict(frame_zoomed, conf=0.7, iou=0.3,device=0)
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

        cv2.putText(annote, "Zoom: "+str(zoom_scale), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("detection",annote)
        key = cv2.waitKey(1)
        if key==32: #space key
            chk=push_data.store()
            print(face_embeddings_test.flag)
            face_embeddings_test.flag=0

        elif key == 27:  # Esc key
            break
        elif key == 13:  # Enter key
            zoom_scale = min(2.0, round(zoom_scale + 0.2,1))
        elif key == 9:  # Tab key
            zoom_scale = max(1.0, round(zoom_scale - 0.2,1))
cap.release()
cv2.destroyAllWindows()