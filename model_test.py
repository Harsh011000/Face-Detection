import cv2
from ultralytics import YOLO


model = YOLO("best.pt")



cv2.setUseOptimized(True)  # Enable optimization
#cv2.setNumThreads(4)       # Set the number of CPU threads
#cv2.ocl.setUseOpenCL(True) # Enable OpenCL (GPU) support
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    true,frame=cap.read()
    if true:
        results = model.predict(frame, conf=0.7, iou=0.3,device=0)

        #for result in results:
        annote=results[0].plot()
        print(results[0].names)

        cv2.imshow("detection",annote)
        if cv2.waitKey(1)==27:
            break;
cap.release()
cv2.destroyAllWindows()