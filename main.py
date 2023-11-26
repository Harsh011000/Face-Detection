from ultralytics import YOLO
import cv2
import face_fetch
import face_embeddings_test
import push_data
import PlayAU as pau

model=YOLO("face-model.pt")

cv2.setUseOptimized(True)  # Enable optimization
zoom_scale = 1.0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Flag to check if the sound is currently playing
pau.sound_playing = False
pau.last_audio_play_time = 0

# # Warning symbol image path
# warning_image_path = 'WARNING_Img.jpg'
# warning_image = cv2.imread(warning_image_path, cv2.IMREAD_UNCHANGED)



while cap.isOpened():
    true,frame=cap.read()
    if true:
        center_x, center_y = int(frame.shape[1] / 2), int(frame.shape[0] / 2)

        # Calculate the cropping region
        crop_width = int(frame.shape[1] / zoom_scale)
        crop_height = int(frame.shape[0] / zoom_scale)

        # Calculate the crop coordinates
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        x2 = min(frame.shape[1], center_x + crop_width // 2)
        y2 = min(frame.shape[0], center_y + crop_height // 2)

        # Crop and resize the frame
        frame_zoomed = frame[y1:y2, x1:x2]
        frame_zoomed = cv2.resize(frame_zoomed, (frame.shape[1], frame.shape[0]))



        results = model.predict(frame_zoomed, conf=0.7, iou=0.3,device=0)
        result=results[0]
        face_lis=[]


        def count_unknown_faces():
            count = 0
            for entry in face_lis:
                if entry[0] == 'unknown':
                    count += 1
            return count


        for box in result.boxes:
            xywh=face_fetch.extract_face(box.xyxy[0].tolist(),frame)
            name=face_embeddings_test.match()
            tmp=[name,xywh]
            face_lis.append(tmp)
        annote=result.plot()
        for x in face_lis:
            print(x)
            cv2.putText(annote,x[0], (int(x[1][0]), int((x[1][1]-50))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)




        # Check if there are more unknown faces
        try:
            # if name == "unknown" and len(face_lis) > 0:
            #     pau.play_audio_threaded()
            # elif name != "unknown" or len(face_lis) < 0:
            #     pau.pygame.mixer.music.stop()
            if count_unknown_faces() > 4 and not pau.sound_playing:
                current_time = pau.time.time()
                if current_time - pau.last_audio_play_time >= 20:
                    pau.play_audio_threaded()
                    pau.sound_playing = True
                    cv2.putText(annote,"Warning",(20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 115 or key == 83:
                        pau.pygame.mixer.music.set_volume(0.0)
                # alpha = 1  # Adjust transparency
                # overlay = cv2.resize(warning_image, (frame.shape[1], frame.shape[0]))
                # frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

            elif count_unknown_faces() <= 4 and pau.sound_playing:
                pau.pygame.mixer.music.stop()
                pau.sound_playing = False
        except:
            pass




        cv2.putText(annote, "Zoom: "+str(zoom_scale), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("detection",annote)
        key = cv2.waitKey(1)
        if key==32: #space key
            chk=push_data.store()
            print(face_embeddings_test.flag)
            face_embeddings_test.flag=0

        elif key == 27:  # Esc key
            break
        elif key == 13:  # Enter key
            zoom_scale = min(2.0, round(zoom_scale + 0.2,1))
        elif key == 9:  # Tab key
            zoom_scale = max(1.0, round(zoom_scale - 0.2,1))
        


cap.release()
cv2.destroyAllWindows()

