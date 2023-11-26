import firebase_admin
from firebase_admin import db, credentials
import face_fetch
from ultralytics import YOLO
import face_embeddings_test
import random
import string


# embd=""
# Nme=""

cred = credentials.Certificate("credentials.json")# generate your own firebase json file
firebase_admin.initialize_app(cred,{"databaseURL":"Enter your db link"})# enter your db link
def store():
    #print(face_embeddings_test.flag)
    #global embd,Nme
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
    # embd=data
    # Nme=name

    numpy_array = data.detach().cpu().numpy()
    python_list = numpy_array.tolist()

    if operation=="p":
        suffix=generate_random_string(5)
        name+=" 0"+suffix
        ref.update({name: python_list})
    elif operation=="u":
        ref.update({name: python_list})
    return True
    # print(face_embeddings_test.flag)
    # face_embeddings_test.show_var()
    # if face_embeddings_test.flag==1:
    #     face_embeddings_test.face_nm.append(Nme)
    #     face_embeddings_test.face_emb.append(embd)
# def update_list():
#     global embd,Nme
#     if embd!="" and Nme!="":
#         return [Nme,embd]
#     else:
#         return ""
# def update_list_fix():
#     global embd,Nme
#     value=update_list()
#     embd=""
#     Nme=""
#     return value
# if __name__=="__main__":
#     store()

def generate_random_string(length=5):
    # Include all digits and symbols except "#"
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))