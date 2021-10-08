import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import data

rep = 0
x_vals = []
y_vals = []
cuenta = 0
maxang = 0
minang = 0
maxv = -1000
maxw = -1000
Base = data.DataBase()
arm = 0

def angle_calculate(a,b,c,first = None,vi = None,ti = None,):
    global maxv, maxw
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
        w = (v-vi)/(tf-ti)
        first = angle
        ti = tf
        vi = v

        if v > maxv:
                maxv = v    
        if w > maxw:
                maxw = w 

        if v >= 1:
            pyautogui.press("P")


    return angle,first,vi,ti,v,w

def image_process (frame,mp_drawing,mp_holistic,holistic,control, first = None,vi = None,ti = None):  
    global angle,maxang, minang, maxangimage, minangimage
    
    #cambios de color y aplicar módulo holistic
    image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    result = holistic.process(image)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        #Landmarks
    try: 
        landmarks = result.pose_landmarks.landmark
        
        if arm == 0:
            #coordenadas de brazo izq
            shoulder_L = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_L = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_L = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            
            angle,first,vi,ti,v,w = angle_calculate(shoulder_L,elbow_L,wrist_L,first,vi,ti)

        else: 
            #coordenadas de brazo der
            shoulder_R = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_R = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_R = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            
            angle,first,vi,ti,v,w = angle_calculate(shoulder_R,elbow_R,wrist_R,first,vi,ti)
        
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
        
        control = game_controller(control)

        if angle > maxang:
                maxang = angle
                maxangimage = image
        elif angle < minang:
            minang = angle
            minangimage = image
          
    except:
        pass
     #dibujar las articulaciones del cuerpo en la imagen
    mp_drawing.draw_landmarks(image, result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (102,31,208),thickness = 2,circle_radius = 3),
                              mp_drawing.DrawingSpec(color = (103,249,237),thickness = 2,circle_radius = 2))
    
    return image,first,vi,ti,control

def game_controller(control):
    global stage, rep
    
    if angle > 100 and control != 0:
        stage = 0
        control = 0
        pyautogui.press("S")
        
    if angle > 70 and angle < 100 and control != 1:
        control = 1
        pyautogui.press(" ")  
        
    if angle < 70 and control != 2:
        control = 2
        pyautogui.press("W")
        if stage == 0:
            stage = 1
            rep += 1
            pyautogui.press("R")  
            print(rep)
    return control

def contador():
    global cuenta
    cuenta += 0.7

def animate(i): 
    try:
        contador()
        x_vals.append(angle)
        y_vals.append(cuenta)

        if len(x_vals) > 20:
            x_vals.pop(0)
            y_vals.pop(0)
        
        plt.cla()
        plt.ylim(0,180)
        plt.ylabel("angle (°)")
        plt.xlabel("time (s)")
        plt.autoscale(enable=True,axis='x')
        plt.plot(y_vals,x_vals,"palevioletred",linewidth=3.0)
    except: pass
             
def Imagen():
    #Definicion de variables
    first = None
    ti = None
    vi = None
    control = None
    
    #data base
    user = Base.get_user()
    
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

def main():
    plt.figure("Angle transition")
    t1 = threading.Thread(target=Imagen, name="t1")
    t1.start()
    ani = FuncAnimation(plt.gcf(),animate,interval=700)

    plt.tight_layout()
    plt.show()
    t1.join()
      
if __name__=="__main__":
    main()