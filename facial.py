import cv2
import face_recognition

image = face_recognition.load_image_file('/home/nx/Facial Recon/Faces/known/Carlos.jpeg')
face_locations = face_recognition.face_locations(image)
# print(face_locations)

# print(cv2.__version__)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for (row1,col1,row2,col2) in face_locations:

    cv2.rectangle(image,(col1,row1),(col2,row2),(0,0,255),2)


cv2.imshow('face',image)
cv2.waitKey(0)
