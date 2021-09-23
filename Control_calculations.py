import cv2 as cv
import mediapipe as mp
import numpy as np
import keyboard
import time

rep = 0
def angle_calculate(a,b,c,first = None,vi = None,ti = None,):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle

    tf = time.perf_counter()

    if first == None:
        first = float(angle)
        ti = tf
        vi=0
        v=0
        w=0

    else:
        v = (angle-first)/(tf-ti)
        w = (v-vi)/tf
        first = angle
        ti = tf
        vi = v
    return angle,first,vi,ti,v,w

def image_process (frame,mp_drawing,mp_holistic,holistic,control, first = None,vi = None,ti = None):  
    global angle
    
    #cambios de color y aplicar módulo holistic
    image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    result = holistic.process(image)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        #Landmarks
    try: 
        landmarks = result.pose_landmarks.landmark
        
        #coordenadas de brazo izq
        shoulder_L = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_L = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_L = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
        
        angle,first,vi,ti,v,w = angle_calculate(shoulder_L,elbow_L,wrist_L,first,vi,ti)
        
        #look angle
        cv.putText(image,str(int(angle)),
                   tuple(np.multiply(elbow_L,[647,510]).astype(int)),
                         cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv.LINE_AA)
        
        #etiquetas
        cv.rectangle(image,(0,0),(210,50),(219,191,255),-1)
        cv.rectangle(image,(410,0),(800,50),(219,191,255),-1)
        
        cv.putText(image,"V. Angular = {:.2f}".format(v),
                   (10,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
        cv.putText(image,"A. Angular = {:.2f}".format(w),
                   (430,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
        
        control = game_controller(angle,control)
          
    except:
        pass
     #dibujar las articulaciones del cuerpo en la imagen
    mp_drawing.draw_landmarks(image, result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (102,31,208),thickness = 2,circle_radius = 3),
                              mp_drawing.DrawingSpec(color = (103,249,237),thickness = 2,circle_radius = 2))
    
    return image,first,vi,ti,control

def game_controller(angle,control):
    global stage, rep
    
    if angle > 110 and control != 0:
        stage = 0
        control = 0
        keyboard.write("s")
        
        
    if angle > 60 and angle < 100 and control != 1:
        control = 1
        keyboard.write(" ")  
        
    if angle < 60 and control != 2:
        control = 2
        keyboard.write("w")
        if stage == 0:
            stage = 1
            rep += 1
            print(rep)
    return control
             
def main():
    #Definicion de variables
    first = None
    ti = None
    vi = None
    control = None

    #setup mediapie
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic 
              
    #Abrir cámara web 
    capture = cv.VideoCapture(0) 
    with mp_holistic.Holistic(min_detection_confidence=0.8,min_tracking_confidence=0.8)as holistic:
        while True:
            #Lerr datos de camara web
            data,frame = capture.read()
            frame = cv.flip(frame,1)
            image,first,vi,ti,control = image_process(frame,mp_drawing,mp_holistic,holistic,control,first,ti,vi)
            cv.imshow('camera',image)
            
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                capture.release() 
                break 
      
if __name__=="__main__":
    main()          