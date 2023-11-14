import firebase_admin
from firebase_admin import db, credentials
import face_fetch
from ultralytics import YOLO
import face_embeddings_test


cred = credentials.Certificate("credentials.json")# generate your own firebase json file
firebase_admin.initialize_app(cred,{"databaseURL":"link of your firebase db"})# enter your db link
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

ref.update({name: python_list})
