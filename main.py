from imutils import face_utils
import dlib
import cv2
from pygame import mixer

thres = 6
mixer.init()
sound = mixer.Sound('alarm.wav')
dlist = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(1)

def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        le_38, le_39, le_41, le_42 = shape[37], shape[38], shape[40], shape[41]
        re_44, re_45, re_47, re_48 = shape[43], shape[44], shape[46], shape[47]
        
        dlist.append((dist(le_38, le_42) + dist(le_39, le_41) + dist(re_44, re_48) + dist(re_45, re_47)) / 4 < thres)
        if len(dlist) > 10:
            dlist.pop(0)
        
        if sum(dlist) >= 4:
            try:
                sound.play()
            except:
                pass
        else:
            try:
                sound.stop()
            except:
                pass
        
    cv2.imshow("Output", image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
