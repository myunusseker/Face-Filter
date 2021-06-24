import face_recognition
import cv2
import numpy as np
import dlib
from math import hypot
from math import atan2
from math import pi
import argparse

def rotate_image(image, angle, filter_name):
    image_center = np.array(image.shape[1::-1]) / 2
    if filter_name == "dog_glass.png":
        image_center = (805,965)
    if filter_name == "glass.png":
        image_center = (962,870)
    if filter_name == "dog_closed.png":
        image_center = (775,915)
    if filter_name == "dog_open.png":
        image_center = (725,846)
    if filter_name == "thug.png":
        image_center = (775,700)
    if filter_name == "ironman.png":
        image_center = (755,835)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--filter', type=str, default='dog_glass', help='filter name options=[dog_glass, glass, dog_closed, dog_open, thug, ironman, bar]')
parser.add_argument('--show_landmarks', type=str, default='false', help='show landmarks if true')
parser.add_argument('--open_mouth', type=str, default='false', help='interactive dog filter')

args = parser.parse_args()

cap = cv2.VideoCapture(0)

filter_name = "%s.png"%args.filter

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

open_mouth = True if args.open_mouth=='true' else False

while True:
    Sucess, img = cap.read()
    img = cv2.flip(img, 1)
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img)
    for face in faces:
        landmarks = predictor(imgGray, face)
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
        right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        up_mouth = (landmarks.part(51).x, landmarks.part(51).y)
        down_mouth = (landmarks.part(57).x, landmarks.part(57).y)
        down_face = (landmarks.part(8).x, landmarks.part(8).y)
        #print((down_mouth[1]-center_nose[1])-(down_face[1]-down_mouth[1]))
      
        if open_mouth:
            if (down_mouth[1]-center_nose[1])-(down_face[1]-down_mouth[1]) > 16:
                filter_name = "dog_open.png"
            else:
                filter_name = "dog_closed.png"
        
        front_filter = cv2.imread('./filters/'+filter_name)
        front_filter = cv2.copyMakeBorder(front_filter, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=[0,0,0,0])
        rotation = -atan2(right_eye[1]-left_eye[1],right_eye[0]-left_eye[0])/pi*180
        if filter_name != "bar.png":
            front_filter = rotate_image(front_filter, rotation, filter_name)
        
        if filter_name == "dog_glass.png":
            nose_width = int(hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])*7.0)
            nose_height = int(nose_width * 0.6708*1.45)
            offset = int(nose_width*30/200)
            top_left = (int(center_nose[0]-nose_width/2)-4,int(center_nose[1]-nose_height/2)-offset)
            bottom_right = (int(center_nose[0]+nose_width/2)-4,int(center_nose[1]+nose_height/2)-offset)

        if filter_name == "glass.png":
            nose_width = int(hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])*3.5)
            nose_height = int(nose_width * 0.6708*1.0)
            offset = int(nose_width*20/200)
            top_left = (int(center_nose[0]-nose_width/2)-0,int(center_nose[1]-nose_height/2)-offset)
            bottom_right = (int(center_nose[0]+nose_width/2)-0,int(center_nose[1]+nose_height/2)-offset)
            
        if filter_name == "thug.png":
            nose_width = int(hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])*4.6)
            nose_height = int(nose_width * 0.6708*1)
            offset = int(nose_width*18/200)
            top_left = (int(center_nose[0]-nose_width/2),int(center_nose[1]-nose_height/2)-offset)
            bottom_right = (int(center_nose[0]+nose_width/2),int(center_nose[1]+nose_height/2)-offset)
            
        if filter_name == "bar.png":
            nose_width = int(hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])*1.6)
            nose_height = int(nose_width * 0.6708*0.5)
            offset = int(nose_width*-150/200)
            top_left = (int(center_nose[0]-nose_width/2),int(center_nose[1]-nose_height/2)-offset)
            bottom_right = (int(center_nose[0]+nose_width/2),int(center_nose[1]+nose_height/2)-offset)
            eye_dist = hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])
            lip_dist = hypot(left_mouth[0]-right_mouth[0], left_mouth[1]-right_mouth[1])
            bar_dist = top_left[0]+5+(bottom_right[0]-top_left[0]-10)*max(0,min((lip_dist/eye_dist-0.60)/0.175,1))
            percent = int(max(0,min((lip_dist/eye_dist-0.60)/0.175,1))*255)
            cv2.rectangle(img, (top_left[0]+5, bottom_right[1] - 25), (int(bar_dist), bottom_right[1]-5), (0, 0+percent, 255-percent), cv2.FILLED)
            cv2.rectangle(img, (top_left[0]+5, bottom_right[1] - 25), (bottom_right[0]-5, bottom_right[1]-5), (255, 255, 255), 2)
            font = cv2.FONT_ITALIC
            cv2.putText(img, 'Smile Meter', (top_left[0]-10, bottom_right[1] - 35), font, 0.8, (255, 255, 255), 1)
            cv2.putText(img, '%.2f%%'%(max(0,min((lip_dist/eye_dist-0.60)/0.175,1))*100.), (top_left[0]+35, bottom_right[1] + 25), font, 0.6, (255, 255, 255), 1)
            continue
        
        if filter_name == "ironman.png":
            nose_width = int(hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])*10)
            nose_height = int(nose_width * 0.6708*1.4)
            offset = int(nose_width*10/200)
            top_left = (int(center_nose[0]-nose_width/2)+5,int(center_nose[1]-nose_height/2)-offset)
            bottom_right = (int(center_nose[0]+nose_width/2)+5,int(center_nose[1]+nose_height/2)-offset)
            
        if filter_name == "dog_closed.png":
            nose_width = int(hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])*7.)
            nose_height = int(nose_width * 0.6708*1.25)
            offset = int(nose_width*22/200)
            top_left = (int(center_nose[0]-nose_width/2)-3,int(center_nose[1]-nose_height/2)-offset)
            bottom_right = (int(center_nose[0]+nose_width/2)-3,int(center_nose[1]+nose_height/2)-offset)
        
        if filter_name == "dog_open.png":
            nose_width = int(hypot(left_eye[0]-right_eye[0], left_eye[1]-right_eye[1])*7.5)
            nose_height = int(nose_width * 0.6708*1.75)
            offset = int(nose_width*2/200)
            top_left = (int(center_nose[0]-nose_width/2)-7,int(center_nose[1]-nose_height/2)-offset)
            bottom_right = (int(center_nose[0]+nose_width/2)-7,int(center_nose[1]+nose_height/2)-offset)
        
        #cv2.circle(front_filter, (775,915), radius=1, color=(0, 255, 0), thickness=15)
        
        ffilter = cv2.resize(front_filter, (nose_width, nose_height))
        filter_gray = cv2.cvtColor(ffilter, cv2.COLOR_BGR2GRAY)
        _, filter_mask = cv2.threshold(filter_gray, 0, 255, cv2.THRESH_BINARY_INV)
        
        filter_area = img[max(0,top_left[1]):min(480,top_left[1]+nose_height),max(0,top_left[0]):min(640,top_left[0]+nose_width)]
        filter_mask = filter_mask[max(0,-top_left[1]):max(0,-top_left[1])+filter_area.shape[0],max(0,-top_left[0]):max(0,-top_left[0])+filter_area.shape[1]]
        
        filter_alpha = cv2.bitwise_and(filter_area, filter_area, mask=filter_mask)
        final_nose = cv2.add(filter_alpha, ffilter[max(0,-top_left[1]):filter_area.shape[0]+max(0,-top_left[1]),max(0,-top_left[0]):max(0,-top_left[0])+filter_area.shape[1]])

        img[max(0,top_left[1]):min(480,top_left[1]+nose_height),max(0,top_left[0]):min(640,top_left[0]+nose_width)] = final_nose
    
    if args.show_landmarks == 'true':
        for i in range(68):
            cv2.circle(img, (landmarks.part(i).x, landmarks.part(i).y), radius=0, color=(0, 0, 255), thickness=5)
    #cv2.circle(img, (center_nose[0],center_nose[1]), radius=1, color=(0, 0, 255), thickness=4)
        
    cv2.imshow("video", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
        
cap.release()
cv2.destroyAllWindows()

