#from ultralytics import YOLO
import cv2
import numpy as np

# model = YOLO("best.pt")
# results=model.predict("image6.jpg",conf=0.7,iou=0.3,device=0,save_crop=True)
# result=results[0]
# box = result.boxes[0]
# cords = box.xywh[0].tolist()
def extract_face(box,frame):
    cords2=box#box.xyxy[0].tolist()
    x=int(cords2[0])
    y=int(cords2[1])
    width=int(int(cords2[2])-int(cords2[0]))
    height=int(int(cords2[3])-int(cords2[1]))
    #print(cords,'\n',cords2)

    original_image = frame#cv2.imread(frame)

    # Define the box coordinates (x, y, width, height)
    #face_box = cords
    #face_box = [int(c) for c in cords]

    # Crop the face region
    #face = original_image[face_box[1]:face_box[1]+face_box[3], face_box[0]:face_box[0]+face_box[2]]
    face = original_image[y:y+height, x-50:x+width+50]

    # Create a new image with the same dimensions as the face
    #new_image = cv2.imread((face_box[2], face_box[3]), dtype=np.uint8)
    #new_image = np.zeros((face_box[2], face_box[3], 3), dtype=np.uint8)

    # Paste the cropped face onto the new image
    #new_image[0:face_box[3], 0:face_box[2]] = face
    new_image = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)

    # Paste the cropped face onto the new image
    new_image[0:face.shape[0], 0:face.shape[1]] = face

    # Save the new image
    cv2.imwrite('run_im.jpg', new_image)
    return [cords2[0],cords2[1],(cords2[2]-cords2[0]),(cords2[3]-cords2[1])]

def extract_face_db(box,frame):
    cords2=box#box.xyxy[0].tolist()
    x=int(cords2[0])
    y=int(cords2[1])
    width=int(int(cords2[2])-int(cords2[0]))
    height=int(int(cords2[3])-int(cords2[1]))
    #print(cords,'\n',cords2)

    original_image = cv2.imread(frame)

    # Define the box coordinates (x, y, width, height)
    #face_box = cords
    #face_box = [int(c) for c in cords]

    # Crop the face region
    #face = original_image[face_box[1]:face_box[1]+face_box[3], face_box[0]:face_box[0]+face_box[2]]
    face = original_image[y:y+height, x-50:x+width+50]

    # Create a new image with the same dimensions as the face
    #new_image = cv2.imread((face_box[2], face_box[3]), dtype=np.uint8)
    #new_image = np.zeros((face_box[2], face_box[3], 3), dtype=np.uint8)

    # Paste the cropped face onto the new image
    #new_image[0:face_box[3], 0:face_box[2]] = face
    new_image = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)

    # Paste the cropped face onto the new image
    new_image[0:face.shape[0], 0:face.shape[1]] = face

    # Save the new image
    cv2.imwrite('push_im.jpg', new_image)
    #return [cords2[0],cords2[1],(cords2[2]-cords2[0]),(cords2[3]-cords2[1])]
