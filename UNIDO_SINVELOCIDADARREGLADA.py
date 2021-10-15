import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from math import dist

rep = 0
x_vals = []
y_vals = []
cuenta = 0
maxang = 0
minang = 0
maxv = -1000
maxw = -1000
#Base = data.DataBase()
x=0
stage=0
angle=0

def angle_calculate(a,b,c):
    global angle
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    

def vw_calculate():
    global maxv, maxw,t,ang,ace,vel,x, angle
    if x==0:
        t=np.array([time.time(),time.time()],dtype=float)
        ang=np.array([0,angle],dtype=float)
        vel=np.array([0,0],dtype=float)
        ace=np.array([0,0],dtype=float)
        x=1
        v=vel[1]
        w=ace[1]
    else:
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
        
        #if v >= 100:
            #pyautogui.press("P")
    return v,w

def game_controller(control):
    global stage, rep, angle,rom,rom_a
    if angle > (rom[1]-rom_a*.4) and control != 0: #abajo
        stage = 0
        control = 0
        print("abajo")
        #pyautogui.press("S")
        
    if angle > (rom[0]+rom_a*.4) and angle < (rom[1]-rom_a*.4) and control != 1: #no movimiento
        control = 1
        print("no mov")
        #pyautogui.press(" ")  
        
    if angle < (rom[0]+rom_a*.4) and control != 2:#arriba
        control = 2
        print("arriba")
        #pyautogui.press("W")
        if stage == 0:
            stage = 1
            rep += 1
            #pyautogui.press("R")  
            print(rep)
    return control

def elbow_coordinate(lm_p,p_arm):
    S= [lm_p[p_arm[0]].x, lm_p[p_arm[0]].y]
    E= [lm_p[p_arm[1]].x, lm_p[p_arm[1]].y]
    W =[lm_p[p_arm[2]].x, lm_p[p_arm[2]].y]
    return S, E , W

def coor_obtain(lm_p, index_list, puntos):
    coor=np.zeros((len(puntos),2))
    #Obtiene coordenadas de puntos específicos de interés
    for i in lm_p:
        for index in index_list:
            if index in puntos:
                x=int(lm_p[index].x*width)
                y=int(lm_p[index].y*height)
                ind=puntos.index(index)
                coor[ind,0]=int(x)
                coor[ind,1]=int(y)
    return coor      

def reemplazar_wrist(mp_drawing,mp_holistic,image,lm,coor):
    global width, height
    #reemplaza corrdenada wrist de pose por la de hands si existe, y dibuja la mano
    coor[0]=int(lm.landmark[mp_holistic.HandLandmark.WRIST].x*width)
    coor[1]=int(lm.landmark[mp_holistic.HandLandmark.WRIST].y*height)
    
    mp_drawing.draw_landmarks(image, lm,mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color = (0,0,0),thickness = 2,circle_radius = 3),
            mp_drawing.DrawingSpec(color =(219,230,101),thickness = 2,circle_radius = 2))
    return image,coor

def maxmin():
    global maxang, minang,angle
    if angle > maxang:
        maxang = angle
        
    elif angle < minang:
        minang = angle

def proporciones(coor):
    prop=np.zeros((1,2))
    espalda=dist((coor[2,0],coor[2,1]),(coor[3,0],coor[3,1])) 
    humero=dist((coor[1,0],coor[1,1]),(coor[2,0],coor[2,1])) 
    tibia=dist((coor[0,0],coor[0,1]),(coor[1,0],coor[1,1]))                               
    prop[0,0]=humero/espalda
    prop[0,1]=tibia/espalda
    return prop

def image_process (frame,mp_drawing,mp_holistic,holistic,control,arm):  
    global maxang, minang,t,coor,width,height 
    #cambios de color y aplicar módulo holistic
    image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    result = holistic.process(image)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    
    puntos=[16,14,12,11,13,15] #todos los puntos de brazos DERECHO-IZQ
    if arm==0:
        p_arm=[11,13,15] # S E W IZQ segun mediapipe, derecho real
        p_cal=[15,13,11,12,1]
    else:
        p_arm=[12,14,16] # S E W DERECHO
        p_cal=[16,14,12,11,1]
    coor=np.zeros((len(puntos),2))
    c=False
        #Landmarks
    try: 
        lm_p = result.pose_landmarks
        lm_lh=result.left_hand_landmarks
        lm_rh=result.right_hand_landmarks

        if lm_p is not None: 
            c=True
            S,E,W=elbow_coordinate(lm_p.landmark, p_arm) #obtenemos coordenadas (S,E,W del brazo a rehabilitar)
            coor=coor_obtain(lm_p.landmark, np.linspace(1,33,33).astype(int).tolist() , puntos) #Obtenemos coordenadas de ambos brazos                 
    except:
        pass

    if c==True:
        if  lm_lh is not None:
            image,coor[len(coor)-1,:]=reemplazar_wrist(mp_drawing, mp_holistic, image, lm_lh, [coor[len(coor)-1,0],coor[len(coor)-1,1]]) 
        if  lm_rh is not None:
            image,coor[0,:]=reemplazar_wrist(mp_drawing, mp_holistic, image, lm_rh,  [coor[0,0],coor[0,1]])


        for i in range(len(coor)-1):
            cv.line(image, (int(coor[i,0]), int(coor[i,1])), (int(coor[i+1,0]), int(coor[i+1,1])), (219,230,101),2)
        for i in range(len(coor)):
            cv.circle(image,(int(coor[i,0]),int(coor[i,1])),3,(102,31,208),2)  

        coor_cal=coor_obtain(lm_p.landmark, np.linspace(1,33,33).astype(int).tolist(), p_cal)
        prp_a=proporciones(coor_cal)
        
        cuadro,image=encuadrar(coor_cal,image)
        if cuadro==True:
            angle_calculate(S, E, W)
            
            v,w=vw_calculate()
            #look angle
            cv.putText(image,str(int(angle)),tuple(np.multiply(E,[647,510]).astype(int)),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv.LINE_AA)
            #etiquetas
            cv.rectangle(image,(0,0),(230,50),(219,191,255),-1)
            cv.rectangle(image,(400,0),(800,50),(219,191,255),-1)
            
            cv.putText(image,"V. Angular = {:.2f}".format(v), (10,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
            cv.putText(image,"A. Angular = {:.2f}".format(w), (410,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
        control = game_controller(control)
        
    return image,control

def colormarco(is_all_true,image):
    if is_all_true==True:
        image = cv.rectangle(image,(0,0),(640,480),(0,255,0),50)
    else:
        image = cv.rectangle(image,(0,0),(640,480),( 3, 3,252),50)  
        cv.putText(image,"Posicionate dentro del recuadro", (10,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
    return image

def encuadrar(coor,image):
    check=np.zeros((1,3))
    #corroborar que no se salga una articulacion de cuadro
    if coor[3,0]>50 and coor[3,0]<width-50 and coor[3,1]>50 and coor[3,1]<height-50: #checar coordenadas de hombro contralaterial
        check[0,0]=True 
    if coor[0,0]<width-50 and coor[0,0]>50 and coor[0,1]<height-50 and coor[0,1]>50: #checar coor de muñeca
        check[0,1]=True
    if coor[4,0]<width-50 and coor[4,0]>50 and coor[4,1]<height-50 and coor[4,1]>50: #checar coor de cabeza
        check[0,2]=True
    is_all_true=np.all((check == True))  
    image=colormarco(is_all_true,image)
    
    return is_all_true, image    

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

def get_image(capture): #ya la puse en el otro
    data,frame = capture.read()
    frame = cv.flip(frame,1)
    return frame

def show_image(image):
    cv.imshow('camera',image)

def Imagen():
    control = 0
    global width, height,capture
    #data base
    #user = Base.get_user()
    width=640
    height=480
    arm=1
    #setup mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic 

    #Abrir cámara web 
    capture = cv.VideoCapture(0) 
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8)as holistic: 
        #aqui debe ir la calibracion
        while True:
            #Leer datos de camara web
            frame=get_image(capture)
            image,control = image_process(frame,mp_drawing,mp_holistic,holistic,control,arm)
            show_image(image)
            
            if cv.waitKey(1) == ord('q'):
                capture.release() 
                cv.destroyAllWindows()  
                break 

def rom_calculate(coor):
    global rom,angle,b
    #min y max angle
    if b==1:
        rom=np.array([360,-360],dtype=float)
    else:
        angle_calculate(coor[2,:],coor[1,:],coor[0,:])
        if angle<rom[0]:
            rom[0]=angle
        if angle>rom[1]:
            rom[1]=angle

def medir_tiempo(cuadro):
    global u,state,a,t,b
    if u==0:
        state=np.array([0,cuadro],dtype=float)
        u=1
        t=0
        a=0
        b=0
    else:
        state=np.flipud(state)
        state[1]=cuadro
        
    if state[0]==False and state[1]==True:
        a=time.time()
        b=1
    if state[0]==True and state[1]==False:
        a=0
        b=0
        t=0
    if np.all(state==True):
        t=time.time() - a
        b=0
    return t

def avisos(image, t):
    global rt
    if t>0 and t<rt[0]:
        cv.putText(image,"Mantente quieto con los brazos extendidos lateralmente", (10,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
    if t>=rt[0] and t<rt[1]:
        cv.putText(image,"Flexiona el codo", (10,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
    if t>=rt[1] and t<rt[2]:
        cv.putText(image,"Extiende el codo", (10,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv.LINE_AA)
    return image

def calibracion(capture,mp_drawing,mp_holistic,holistic,puntos):  
    global width,height
    global prp, rt
    global rom
    rom=np.array([360,-360],dtype=float) 
    rt=[5,15,30]
    r=True
    while r==True:
        frame=get_image(capture)
        image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        result = holistic.process(image)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        try: 
            lm_p = result.pose_landmarks
            if lm_p is not None:
                coor=coor_obtain(lm_p.landmark, np.linspace(1,33,33).astype(int).tolist(), puntos)
                cuadro,image=encuadrar(coor,image) #Si cuadro es true, las coordenadas están bien posicionadas
                t_verde=medir_tiempo(cuadro)
                if t_verde>=rt[0]:
                    rom_calculate(coor)
                    if t_verde>=rt[2]:
                        r=False
                image=avisos(image,t_verde)
            else:
                image=colormarco(False,image)
        except:
            pass
        show_image(image)
        if cv.waitKey(1) == ord('q'):
            capture.release() 
            cv.destroyAllWindows()  
            break 
        
    #return prop

def cal():
    global width, height,u,capture
    #data base
    #user = Base.get_user()
    arm=1
    u=0
    width=640
    height=480
    #setup mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic 
    capture = cv.VideoCapture(0) 
    with mp_holistic.Holistic(min_detection_confidence=0.8,min_tracking_confidence=0.8) as holistic:
        if arm==0: #el orden es: W,E,S,SS,H, head
            puntos=[15,13,11,12,1]#derecho
        else:
            puntos=[16,14,12,11,1]#izquierdo
        #Aqui debe ir la calibracion    
        calibracion(capture, mp_drawing, mp_holistic, holistic, puntos)
            

def main():
    global rom,rom_a
    cal()
    rom_a=rom[1]-rom[0]
    print(rom_a)
    plt.figure("Angle transition")
    t1 = threading.Thread(target=Imagen, name="t1")
    t1.start()
    ani = FuncAnimation(plt.gcf(),animate,interval=700)

    plt.tight_layout()
    plt.show()
    t1.join()

if __name__=="__main__":
    main()