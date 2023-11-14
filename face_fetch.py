import cv2
import numpy as np


def extract_face(box,frame):
    cords2=box#box.xyxy[0].tolist()
    x=int(cords2[0])
    y=int(cords2[1])
    width=int(int(cords2[2])-int(cords2[0]))
    height=int(int(cords2[3])-int(cords2[1]))

    original_image = frame#cv2.imread(frame)



    # Crop the face region
    face = original_image[y:y+height, x:x+width]



    # Paste the cropped face onto the new image
    new_image = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)

    # Paste the cropped face onto the new image
    new_image[0:face.shape[0], 0:face.shape[1]] = face

    # Save the new image
    cv2.imwrite('run_im.jpg', new_image)
    return [cords2[0],cords2[1],(cords2[2]-cords2[0]),(cords2[3]-cords2[1])]

def extract_face_db(box,frame):
    cords2=box
    x=int(cords2[0])
    y=int(cords2[1])
    width=int(int(cords2[2])-int(cords2[0]))
    height=int(int(cords2[3])-int(cords2[1]))
    original_image = cv2.imread(frame)

    # Crop the face region
    face = original_image[y:y+height, x:x+width]

    # Paste the cropped face onto the new image
    new_image = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)

    # Paste the cropped face onto the new image
    new_image[0:face.shape[0], 0:face.shape[1]] = face

    # Save the new image
    cv2.imwrite('push_im.jpg', new_image)