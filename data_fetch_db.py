import firebase_admin
from firebase_admin import db, credentials
import numpy as np
import torch

def get_data(data):
    if not firebase_admin._apps:
        cred = credentials.Certificate("credentials.json")# generate your own firebase json file
        firebase_admin.initialize_app(cred,{"databaseURL":"Enter your db link"})# enter your db link
    ref = db.reference("/")
    records=ref.get()
    names=list((records.keys()))
    names_=[]
    vectors=[]
    for x in names:
        embd=db.reference("/"+x).get()
        x_=(x.split(" 0"))[0]
        names_.append(x_)
        numpy_array = np.array(embd)
        tensor_data = torch.tensor(numpy_array).to(data.device)
        vectors.append(tensor_data)

    return [names_,vectors]