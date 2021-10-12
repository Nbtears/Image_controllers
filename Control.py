import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import Base


rep = 0
x_vals = []
y_vals = []
cuenta = 0
maxang = 0
minang = 0
maxv = -1000
maxw = -1000
x=0
stage=0
puntos=[16,14,12,11,13,15]
coor=np.zeros((len(puntos),2))
index_list=np.linspace(1,33,33).astype(int).tolist() 

def angle_calculate(a,b,c):
    global maxv, maxw,t,ang,ace,vel,x
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    if x==0:
        t=np.array([time.time(),time.time()],dtype=float)
        ang=np.array([0,angle],dtype=float)
        vel=np.array([0,0],dtype=float)
        ace=np.array([0,0],dtype=float)
        x=1
        v=vel[1]
        w=ace[1]
    else:
        #print("entro")
        ang=np.flipud(ang)
        ang[1]=angle
        t=np.flipud(t)
        t[1]=time.time()
        vel=np.flipud(vel)
        vel[1]=(ang[1]-ang[0])/(t[1]-t[0])
        ace=np.flipud(ace)
        ace[1]=(vel[1]-vel[0])/(t[1]-t[0])
        v=vel[1]
        w=ace[1]

        if v > maxv:
                maxv = v    
        if w > maxw:
                maxw = w 
        
        if v >= 100:
            pyautogui.press("P")


    return angle,v,w
    
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

def image_process (frame,mp_drawing,mp_holistic,holistic,control):  
    global angle,maxang, minang, maxangimage, minangimage,t,ang,coor,width,height
    #cambios de color y aplicar módulo holistic
    image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    result = holistic.process(image)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    c=False
        #Landmarks
    try: 
        
        lm_p = result.pose_landmarks.landmark
        lm_lh=result.left_hand_landmarks
        lm_rh=result.right_hand_landmarks
        
        #coordenadas de brazo derecho
        S= [lm_p[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                lm_p[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
        E= [lm_p[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                lm_p[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
        W = [lm_p[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                lm_p[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

        #coordenadas brazo izq
        H= [lm_p[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                lm_p[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
        C= [lm_p[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                lm_p[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
        M = [lm_p[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                lm_p[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]

        if arm[0] == 'R':
            angle,v,w = angle_calculate(S,E,W)
             #look angle
            cv.putText(image,str(int(angle)),tuple(np.multiply(E,[647,510]).astype(int)),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv.LINE_AA)
        else:
            angle,v,w = angle_calculate(H,C,M)
             #look angle
            cv.putText(image,str(int(angle)),tuple(np.multiply(C,[647,510]).astype(int)),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv.LINE_AA)
        
        if lm_p is not None:
            c=True
            for i in lm_p:
                for index in index_list:
                    if index in puntos:
                        x=int(lm_p[index].x*width)
                        y=int(lm_p[index].y*height)
                        ind=puntos.index(index)
                        coor[ind,0]=int(x)
                        coor[ind,1]=int(y)
                             
    except:
        pass
    
    if c==True:
        if  lm_lh is not None:
            coor[len(coor)-1,0]=int(lm_lh.landmark[mp_holistic.HandLandmark.WRIST].x*width)
            coor[len(coor)-1,1]=int(lm_lh.landmark[mp_holistic.HandLandmark.WRIST].y*height)
            
            mp_drawing.draw_landmarks(image, lm_lh,mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color = (0,0,0),thickness = 2,circle_radius = 3),
                mp_drawing.DrawingSpec(color =(219,230,101),thickness = 2,circle_radius = 2))  
        
        if  lm_rh is not None:
            coor[0,0]=int(lm_rh.landmark[mp_holistic.HandLandmark.WRIST].x*width)
            coor[0,1]=int(lm_rh.landmark[mp_holistic.HandLandmark.WRIST].y*height)
            
            mp_drawing.draw_landmarks(image, lm_rh, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color = (102,31,208),thickness = 2,circle_radius = 3),
                mp_drawing.DrawingSpec(color = (219,230,101),thickness = 2,circle_radius = 2)) 

        for i in range(len(coor)-1):
            cv.line(image, (int(coor[i,0]), int(coor[i,1])), (int(coor[i+1,0]), int(coor[i+1,1])), (219,230,101),2)
        for i in range(len(coor)):
            cv.circle(image,(int(coor[i,0]),int(coor[i,1])),3,(102,31,208),2)  
        
        #etiquetas
        cv.rectangle(image,(0,0),(230,50),(219,191,255),-1)
        cv.rectangle(image,(400,0),(800,50),(219,191,255),-1)
        
        cv.putText(image,"V. Angular = {:.2f}".format(v), (10,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
        cv.putText(image,"A. Angular = {:.2f}".format(w), (410,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)

        control = game_controller(control)
        if angle > maxang:
                maxang = angle
                maxangimage = image
        elif angle < minang:
            minang = angle
            minangimage = image
        
    return image,control

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

        if rep >= 5:
            plt.close()
    except: 
        pass

def Imagen():
    control = 0
    global width, height, user    
    #setup mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic 

    #Abrir cámara web 
    capture = cv.VideoCapture(0) 
    with mp_holistic.Holistic(min_detection_confidence=0.8,min_tracking_confidence=0.8)as holistic:
        while True:
            #Leer datos de camara web
            data,frame = capture.read()
            height,width,_=frame.shape
            frame = cv.flip(frame,1)
            image,control = image_process(frame,mp_drawing,mp_holistic,holistic,control)
            cv.imshow('camera',image)

            if rep >= 5 :
                capture.release() 
                cv.destroyAllWindows()  
                write_base()
                break 
            
            if cv.waitKey(1) == ord('q'):
                capture.release() 
                cv.destroyAllWindows()  
                write_base()
                break 

def base_control():
    global Db,arm,user
    Db = Base.DataBase()
    user = Db.get_user()
    arm = Db.show_arm(user[1])

def write_base():
    Db.insert_data(maxang,minang,maxv,maxw,0,0,rep,1,user[0])

def main():
    base_control()
    plt.figure("Angle transition")
    t1 = threading.Thread(target=Imagen, name="t1")
    t1.start()
    ani = FuncAnimation(plt.gcf(),animate,interval=700)

    plt.tight_layout()
    plt.show()
    t1.join()

if __name__=="__main__":
    main()