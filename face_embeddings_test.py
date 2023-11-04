import face_recognition as fr
import cv2
import numpy as np

face_emb=[]
face_nm=[]
def match():
    face_image=cv2.imread('run_im.jpg')
    face_image = cv2.resize(face_image, (256, 256)) #I resize that picture to 256x256 as I saw it was giving better results
    face_encodings = fr.face_encodings(face_image)
    if len(face_encodings) > 0:
        face_embedding_un = face_encodings[0]
        if len(face_emb) == 0 and len(face_nm) == 0:
            face_image2 = cv2.imread('test_im1.jpg')
            face_image2 = cv2.resize(face_image2, (256, 256)) #I resize that picture to 256x256 as I saw it was giving better results
            face_embedding2 = fr.face_encodings(face_image2)[0] #This converts the face into a face embedding, a 128-d vector.
            face_emb.append(face_embedding2)
            face_nm.append("Harsh")

            face_image3 = cv2.imread('test_im3.jpg')
            face_image3 = cv2.resize(face_image3, (256, 256))  # I resize that picture to 256x256 as I saw it was giving better results
            face_embedding3 = fr.face_encodings(face_image3)[0]  # This converts the face into a face embedding, a 128-d vector.
            face_emb.append(face_embedding3)
            face_nm.append("Sirshak")

            face_image3 = cv2.imread('test_im7.jpg')
            face_image3 = cv2.resize(face_image3, (256, 256))  # I resize that picture to 256x256 as I saw it was giving better results
            face_embedding3 = fr.face_encodings(face_image3)[0]  # This converts the face into a face embedding, a 128-d vector.
            face_emb.append(face_embedding3)
            face_nm.append("Himesh")

            name=find_match(face_embedding_un)
            return name
        else:
            name = find_match(face_embedding_un)
            return name
    else:
        return "unknown"

def find_match(unknown):
    unknown_face_embedding = unknown

    # Calculate the Euclidean distances between the unknown face and each known face
    distances = [np.linalg.norm(known_emb - unknown_face_embedding) for known_emb in face_emb]

    # Find the index of the known face embedding with the minimum distance
    min_distance_index = np.argmin(distances)

    # Set a threshold for face recognition
    threshold = 0.6  # You can adjust this value as needed

    # Check if the minimum distance is less than the threshold to determine recognition
    if distances[min_distance_index] <= threshold:
        #recognized = True
        return face_nm[min_distance_index]
    else:
        #recognized = False
        return "unknown"


# face_embedding = fr.face_encodings(face_image)[0] #This converts the face into a face embedding, a 128-d vector.
# #x.append(face_embedding) #add the face embedding to training data
# #y.append(idx) #add class number to training labels
# print(face_embedding.shape)

# face_image2=cv2.imread('test_im4.jpg')
# face_image2 = cv2.resize(face_image2, (256, 256)) #I resize that picture to 256x256 as I saw it was giving better results
# face_embedding2 = fr.face_encodings(face_image2)[0] #This converts the face into a face embedding, a 128-d vector.
# #x.append(face_embedding) #add the face embedding to training data
# #y.append(idx) #add class number to training labels
# print(face_embedding2.shape)

# Generate two random embeddings
# emb1 = np.random.rand(128)
# emb2 = np.random.rand(128)

# Calculate euclidean distance between embeddings
# dist = np.linalg.norm(face_embedding - face_embedding2) #good
# #dist = np.linalg.norm(embedding - embedding3)
#
# # Define threshold for similarity
# thresh = 0.6
#
# if dist < thresh:
#     print("Same person")
# else:
#     print("Different persons")

import face_recognition
# Load the face image
# face_image = cv2.imread('test_im4.jpg')
#
# # Convert the image to grayscale
# face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#
# # Apply histogram equalization for contrast enhancement
# face_image_gray = cv2.equalizeHist(face_image_gray)
#
# # Reduce noise using Gaussian blur
# face_image_gray = cv2.GaussianBlur(face_image_gray, (5, 5), 0)
#
# # Resize the image to a consistent size
# face_image_gray = cv2.resize(face_image_gray, (256, 256))
#
# # Normalize pixel values to the range [0, 1]
# face_image_normalized = (face_image_gray - np.min(face_image_gray)) / (np.max(face_image_gray) - np.min(face_image_gray))
#
# # Convert the grayscale image to 8-bit RGB format
# face_image_rgb = cv2.cvtColor(face_image_normalized, cv2.COLOR_GRAY2BGR)
#
# # Calculate the face encoding
# my_face_encoding = face_recognition.face_encodings(face_image_rgb)[0]
# picture_of_me = face_recognition.load_image_file("test_im6.jpg")
# my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

# unknown_picture = face_recognition.load_image_file("run_im.jpg")
# unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
#
# # Now we can see the two face encodings are of the same person with `compare_faces`!
#
# results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
#
# if results[0] == True:
#     print("It's a picture of me!")
# else:
#     print("It's not a picture of me!")
# print(results)