#import cv2
#import face_recognition
#import os

import face_recognition
from cv2 import cv2
import numpy as np
import os


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#codec = 0x47504A4D  # MJPG
#video_capture.set(cv2.CAP_PROP_FPS, 30.0)
#video_capture.set(cv2.CAP_PROP_FOURCC, codec)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

faces_path = '/Users/maytham/Desktop/Facial Recon/Faces/known/'

known_face_encodings = []
known_face_names = []

for img in os.listdir(faces_path):
    print(img)
    print(img[0])
    if img[0] == ".":
        pass
    else:
        
        name = img[:-5]

        path = "/Users/maytham/Desktop/Facial Recon/Faces/known/{}".format(img)
        new_image = face_recognition.load_image_file(path)
        new_image_encoding = face_recognition.face_encodings(new_image)[0]

        known_face_encodings.append(new_image_encoding)
        known_face_names.append(name)



# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
img_counter = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    #if not ret:
    #    print("failed to grab frame")
    #    break


    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]




    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        #face_landmarks_list = face_recognition.face_landmarks#(rgb_small_frame)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    
    #cv2.imshow("test", rgb_small_frame)
    #print(face_locations)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif (k%256 == 32) & (face_locations != []):
        # SPACE pressed
        print(name)
        if name != "Unknown":
            print("Person already exists!")
            
        else:
        #if True:
            top = face_locations[0][0]      *4
            right = face_locations[0][1]    *4
            bottom = face_locations[0][2]   *4
            left = face_locations[0][3]     *4


            cropped = frame[top:bottom, left:right]

            name = input("Enter your name: ")
            path = '/Users/maytham/Desktop/Facial Recon/Faces/known/'

            #img_name = "opencv_frame_{}.png".format(img_counter)
            img_name = "{}.jpeg".format(name)

            cv2.imwrite(os.path.join(path, img_name), cropped)
            print("{} written!".format(img_name[:-5]))
            img_counter += 1

            path = "/Users/maytham/Desktop/Facial Recon/Faces/known/{}.jpeg".format(name)
            new_image = face_recognition.load_image_file(path)
            new_image_encoding = face_recognition.face_encodings(new_image)[0]

            #print("encoding: ", new_image_encoding)

            known_face_encodings.append(new_image_encoding)
            known_face_names.append(name)

    elif (k%256 == 32) & (face_locations == []):
        print("No ghosts allowed!")



    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()