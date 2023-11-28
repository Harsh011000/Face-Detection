import firebase_admin
from firebase_admin import db, credentials
import face_fetch
from ultralytics import YOLO
import face_embeddings_test
import random
import string




cred = credentials.Certificate("credentials.json")# generate your own firebase json file
firebase_admin.initialize_app(cred,{"databaseURL":"Database link"})# enter your db link
def store():

    operation=str(input("press 'p' to push / 'u' to update data"))
    name=str(input("Enter your name: "))
    ref = db.reference("/")

    image="sample.jpg" #input the image name you want to store in db
    model=YOLO("face-model.pt")
    results=model.predict(image,conf=0.7,iou=0.3,device=0)
    result=results[0]
    box=result.boxes[0]
    face_fetch.extract_face_db(box.xyxy[0].tolist(),image)
    data=face_embeddings_test.extract_embedding("push_im.jpg")


    numpy_array = data.detach().cpu().numpy()
    python_list = numpy_array.tolist()

    if operation=="p":
        suffix=generate_random_string(5)
        name+=" 0"+suffix
        ref.update({name: python_list})
    elif operation=="u":
        ref.update({name: python_list})
    return True


def generate_random_string(length=5):
    # Include all digits and symbols except "#"
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))