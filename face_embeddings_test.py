import numpy as np
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
import torch
import data_fetch_db
#import push_data

face_emb=[]
face_nm=[]
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet = resnet.to("cuda" if torch.cuda.is_available() else "cpu")
flag=0

def load_image(filename):
    image = Image.open(filename)
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image to the required input size (160x160)
    image = image.resize((160, 160), Image.LANCZOS)
    return np.array(image)

def extract_embedding(filename):
    pixels = load_image(filename)
    # Preprocess the image by subtracting the mean and scaling
    pixels = fixed_image_standardization(pixels)
    # Convert to tensor and move to GPU if available
    pixels = torch.tensor(pixels).to("cuda" if torch.cuda.is_available() else "cpu")
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).float()
    # Calculate the embedding
    embedding = resnet(pixels)
    return embedding

def match():
    face_embd=extract_embedding('run_im.jpg')
    if len(face_embd) > 0:
        face_embedding_un = face_embd
        global flag
        if flag==0:
            tmp=data_fetch_db.get_data(face_embedding_un)
            global face_nm,face_emb
            face_nm=tmp[0]
            face_emb=tmp[1]
            print(face_nm)
            print(face_emb)
            print(len(face_nm),len(face_emb))
            flag=1

        # global face_nm, face_emb
        # store=push_data.update_list_fix()
        # if store!="":
        #     face_nm.append(store[0])
        #     face_emb.append(store[1])
        name=find_match(face_embedding_un)
        return name

    else:
        return "unknown"

def find_match(unknown):
    unknown_face_embedding = unknown

    # Calculate the Euclidean distances between the unknown face and each known face
    distances = [(known_emb - unknown_face_embedding).norm().item() for known_emb in face_emb]

    # Find the index of the known face embedding with the minimum distance
    min_distance_index = np.argmin(distances)

    # Set a threshold for face recognition
    threshold = 0.7  # You can adjust this value as needed

    # Check if the minimum distance is less than the threshold to determine recognition
    if distances[min_distance_index] <= threshold:
        #recognized = True
        return face_nm[min_distance_index]
    else:
        #recognized = False
        return "unknown"
# def show_var():
#     global face_nm,face_emb
#     print(face_nm)
#     print(face_emb)