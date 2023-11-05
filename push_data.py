import firebase_admin
from firebase_admin import db, credentials
import face_fetch
from ultralytics import YOLO
import face_embeddings_test
import numpy as np
import torch

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred,{"databaseURL":"https://visionai-4e3eb-default-rtdb.asia-southeast1.firebasedatabase.app/"})
name=str(input("Enter your name: "))
ref = db.reference("/")
#print(type(ref.get()))
#ref.update({"hello":"world"})
#ref.delete()
image="sample.jpg"
model=YOLO("best.pt")
results=model.predict(image,conf=0.7,iou=0.3,device=0)
result=results[0]
box=result.boxes[0]
face_fetch.extract_face_db(box.xyxy[0].tolist(),image)
data=face_embeddings_test.extract_embedding("push_im.jpg")
#print(data)
numpy_array = data.detach().cpu().numpy()
python_list = numpy_array.tolist()
#ref.update({"test_data":data})
ref.update({name: python_list})
# data_from_database = db.reference("/"+name).get()#ref.get()  # Replace with the actual fetched data
#
# # Convert the Python list to a NumPy array
# numpy_array = np.array(data_from_database)
#
# # Convert the NumPy array to a PyTorch tensor
# tensor_data = torch.tensor(numpy_array).to(data.device)
# print(tensor_data)
# face_emb=[]
# face_emb.append(data)
# print(face_embeddings_test.find_match(tensor_data))