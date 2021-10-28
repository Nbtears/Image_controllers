import sys
import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from math import dist
#Gimport Base
n=0
rep = 0
x_vals = []
y_vals = []
cuenta = 0
maxang = -360
minang = 360
maxv = -1000
maxw = -1000
vtotal=0
wtotal=0
angtotal=0
x = 0
stage = 0
angle = 0
j=0
s=0


def angle_calculate(a, b, c):
    global angle
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle


def vw_calculate():
    global maxv, maxw, t, ang, ace, vel, x, angle,j,vtotal,wtotal,angtotal
    if x == 0:
        t = np.array([time.time(), time.time()], dtype=float)
        ang = np.array([0, angle], dtype=float)
        vel = np.array([0, 0], dtype=float)
        ace = np.array([0, 0], dtype=float)
        x = 1
        v = vel[1]
        w = ace[1]
    else:
        j+=1
        ang = np.flipud(ang)
        ang[1] = angle
        t = np.flipud(t)
        t[1] = time.time()
        vel = np.flipud(vel)
        vel[1] = (ang[1]-ang[0])/(t[1]-t[0])
        ace = np.flipud(ace)
        ace[1] = (vel[1]-vel[0])/(t[1]-t[0])
        v = vel[1]
        w = ace[1]
        if v > maxv:
            maxv = v
        if w > maxw:
            maxw = w

        angtotal+=angle
        vtotal+=v
        wtotal+=w
        # if v >= 100:
            # pyautogui.press("P")
    return v, w


def game_controller(control):
    global stage, rep, angle, rom, rom_a
    if angle > (rom[1]-rom_a*.4) and control != 0:  # abajo
        stage = 0
        control = 0
        pyautogui.press("S")

    if angle > (rom[0]+rom_a*.4) and angle < (rom[1]-rom_a*.4) and control != 1:  # no movimiento
        control = 1
        pyautogui.press(" ")

    if angle < (rom[0]+rom_a*.4) and control != 2:  # arriba
        control = 2
        pyautogui.press("W")
        if stage == 0:
            stage = 1
            rep += 1
            pyautogui.press("R")
            print(rep)
    return control


def elbow_coordinate(lm_p, p_arm):
    S = [lm_p[p_arm[2]].x, lm_p[p_arm[2]].y]
    E = [lm_p[p_arm[1]].x, lm_p[p_arm[1]].y]
    W = [lm_p[p_arm[0]].x, lm_p[p_arm[0]].y]
    return S, E, W


def coor_obtain(lm_p, index_list, puntos):
    coor = np.zeros((len(puntos), 2))
    # Obtiene coordenadas de puntos específicos de interés
    for i in lm_p:
        for index in index_list:
            if index in puntos:
                x = int(lm_p[index].x*width)
                y = int(lm_p[index].y*height)
                ind = puntos.index(index)
                coor[ind, 0] = int(x)
                coor[ind, 1] = int(y)
    return coor


def reemplazar_wrist(mp_drawing, mp_holistic, image, lm, coor):
    global width, height
    # reemplaza corrdenada wrist de pose por la de hands si existe, y dibuja la mano
    coor[0] = int(lm.landmark[mp_holistic.HandLandmark.WRIST].x*width)
    coor[1] = int(lm.landmark[mp_holistic.HandLandmark.WRIST].y*height)

    mp_drawing.draw_landmarks(image, lm, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(219, 230, 101), thickness=2, circle_radius=2))
    return image, coor


def maxmin():
    global maxang, minang, angle
    if angle > maxang:
        maxang = angle

    elif angle < minang:
        minang = angle


def proporciones(coor):
    prop = np.zeros((1, 2))
    espalda = dist((coor[2, 0], coor[2, 1]), (coor[3, 0], coor[3, 1]))
    humero = dist((coor[1, 0], coor[1, 1]), (coor[2, 0], coor[2, 1]))
    tibia = dist((coor[0, 0], coor[0, 1]), (coor[1, 0], coor[1, 1]))
    prop[0, 0] = humero/espalda
    prop[0, 1] = tibia/espalda
    return prop


def image_process(frame, holistic, control,exer):
    global t, coor, mp_drawing, mp_holistic, fvw, coor_mano
    # cambios de color y aplicar módulo holistic
    image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    result = holistic.process(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    coor = np.zeros((len(puntos), 2))
    coor_mano=np.zeros((len(p_mano), 2))
    c = False

    # Landmarks
    try:
        lm_p = result.pose_landmarks
        if arm == 1:
            lm = result.right_hand_landmarks
        else:
            lm = result.left_hand_landmarks

        if lm_p is not None:
            c = True
            # obtenemos coordenadas (S,E,W del brazo a rehabilitar)
            S, E, W = elbow_coordinate(lm_p.landmark, puntos[0:3])
            coor = coor_obtain(lm_p.landmark, np.linspace(1, 33, 33).astype(
                int).tolist(), puntos)    
    except:
        pass

    if c == True:
        if lm is not None:
            coor_mano=coor_obtain(lm.landmark, np.linspace(1, 20, 20).astype(
                int).tolist(), p_mano)
            image, coor[0, :] = reemplazar_wrist(
                mp_drawing, mp_holistic, image, lm, [coor[0, 0], coor[0, 1]])

        #dibujar los semgentos y articulaciones
        for i in range(len(coor)-2):
            cv.line(image, (int(coor[i, 0]), int(coor[i, 1])), (int(
                coor[i+1, 0]), int(coor[i+1, 1])), (219, 230, 101), 2)
        for i in range(len(coor)-1):
            cv.circle(image, (int(coor[i, 0]), int(
                coor[i, 1])), 3, (102, 31, 208), 2)

        cuadro, image = encuadrar(coor, image)

        if cuadro == True:
            if exer==1: #ejercicio de flexion-extension
                angle_calculate(S, E, W)
                if fvw == 2:
                    vw_calculate()
                    fvw = 0
                fvw += 1
                maxmin()
                # look angle
                cv.putText(image, str(int(angle)), tuple(np.multiply(E, [647, 510]).astype(
                    int)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
                # etiquetas
                cv.rectangle(image, (0, 0), (230, 50), (219, 191, 255), -1)
                cv.rectangle(image, (400, 0), (800, 50), (219, 191, 255), -1)

                cv.putText(image, "V. Angular = {:.2f}".format(
                    vel[1]), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
                cv.putText(image, "A. Angular = {:.2f}".format(
                    ace[1]), (410, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
                cv.putText(image, "Flexion y Extension", (225, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
                control = game_controller(control)
            else:
                #ejercicio de supinación
                image=sup_pro(coor_mano,E,W,image)  
                cv.putText(image, "Supinacion y pronacion", (225, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
                
    return image, control

def sup_pro(coor_mano,E,W,image):
    global sp_vec, rep_sp,s
    if s==0:
        rep_sp=0
        sp_vec = np.array([-1, -1], dtype=float)
        s=1
    else:
        if coor_mano[0,0]>coor_mano[1,0]: #si pulgar está a la derecha
            sp=0
        else:
            sp=1
        sp_vec = np.flipud(sp_vec)
        sp_vec[1]=sp
        
        m=pendiente(E,W)
        if (m>4 or m<-4):
            if np.sum(sp_vec)==1:
                rep_sp+=1
                print(rep_sp)
        else:
            image = cv.rectangle(image, (0, 0), (640, 480), (3, 3, 252), 50)
            cv.putText(image, "Posiciona tu antebrazo verticalmente",(135, 470), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
        
    return image

def pendiente(E,W):
    try:
        m=(W[1]-E[1])/(W[0]-E[0])
    except:
        m=1000
    return m
    
def colormarco(is_all_true, image):
    if is_all_true == True:
        image = cv.rectangle(image, (0, 0), (640, 480), (0, 255, 0), 50)
    else:
        image = cv.rectangle(image, (0, 0), (640, 480), (3, 3, 252), 50)
        cv.putText(image, "Posicionate dentro del recuadro y en medio de la linea vertical",
            (10, 470), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
    return image


def encuadrar(coor, image):
    global width
    check = np.zeros((4, 1))
    # corroborar que no se salga una articulacion de cuadro
    # checar coordenadas de hombro contralaterial
    if coor[3, 0] > 50 and coor[3, 0] < width-50 and coor[3, 1] > 50 and coor[3, 1] < height-50:
        check[0] = True
    if coor[0, 0] < width-50 and coor[0, 0] > 50 and coor[0, 1] < height-50 and coor[0, 1] > 50:  # checar coor de muñeca
        check[1] = True
    if coor[4, 0] < width-50 and coor[4, 0] > 50 and coor[4, 1] < height-50 and coor[4, 1] > 50:  # checar coor de cabeza
        check[2] = True
    if coor[1, 0] < width-50 and coor[1, 0] > 50 and coor[1, 1] < height-50 and coor[1, 1] > 50:  # checar coor de codo
        check[3] = True
    is_all_true = np.all((check == True))

    if arm == 0:
        vx = 200
        if coor[3, 0] < vx and coor[2, 0] > vx and is_all_true == True:
            pi = True
        else:
            pi = False
            cv.line(image, (vx, 100), (vx, 400),
                    (219, 230, 101), 3)  # vertical
    else:
        vx = width-200
        if coor[2, 0] < vx and coor[3, 0] > vx and is_all_true == True:
            pi = True
        else:
            pi = False
            cv.line(image, (vx, 100), (vx, 400),(219, 230, 101), 3)  # vertical

    image = colormarco(pi, image)

    return pi, image


def contador():
    global cuenta
    cuenta += 0.7


def animate(i):
    global capture,current_time,tlim,n,j,rep,rep_sp
    try:
        contador()
        x_vals.append(angle)
        y_vals.append(cuenta)
        if len(x_vals) > 20:
            x_vals.pop(0)
            y_vals.pop(0)

        plt.cla()
        plt.ylim(0, 180)
        plt.ylabel("Angulo (°)")
        plt.xlabel("Tiempo (s)")
        plt.autoscale(enable=True, axis='x')
        plt.plot(y_vals, x_vals, "palevioletred", linewidth=3.0)

        if current_time>=(tlim/2):
            pyautogui.press("H")
            plt.close()

        if current_time>=tlim and n==0:
            n=1;
            #write_base()
            print(angtotal/j)
            print(vtotal/j)
            print(wtotal/j)
            cv.destroyWindow('camera')
            capture.release()
            sys.exit(0)

    except:
        pass


def get_image(capture):
    data, frame = capture.read()
    frame = cv.flip(frame, 1)
    return frame


def show_image(image):
    cv.imshow('camera', image)


def Imagen():
    global fvw, width, height, capture, mp_drawing, mp_holistic,current_time,n
    fvw = 2 #cada cuantos frames se calculará v y w
    control = 0
    inicio=time.time()
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while n==0:
            current_time=time.time()-inicio
            if current_time<=(tlim/2):
                exer=1
            else:
                exer=2
            frame = get_image(capture)
            image, control = image_process(frame, holistic, control,exer)
            show_image(image)

            if cv.waitKey(1) == ord('q'):
                capture.release()
                cv.destroyAllWindows()
                break
        

def rom_calculate(coor):
    global rom, angle, b
    angle_calculate(coor[2, :], coor[1, :], coor[0, :])
    if angle < rom[0]:
        rom[0] = angle
    if angle > rom[1]:
        rom[1] = angle


def medir_tiempo(cuadro):
    global u, state, a, t, b, rom
    if u == 0:
        state = np.array([0, cuadro], dtype=float)
        u = 1
        t = 0
        a = 0
        b = 0
    else:
        state = np.flipud(state)
        state[1] = cuadro

    if state[0] == False and state[1] == True:
        a = time.time()
        b = 1
        rom = np.array([360, -360], dtype=float)
    if state[0] == True and state[1] == False:
        a = 0
        b = 0
        t = 0
        rom = np.array([360, -360], dtype=float)
    if np.all(state == True):
        t = time.time() - a
        b = 0
    return t


def avisos(image, t):
    global rt
    if t > 0 and t < rt[0]:
        cv.putText(image, " Mantente quieto", (235, 470),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
    if t >= rt[0] and t < rt[1]:
        cv.putText(image, "Flexiona el codo", (235, 470),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
    if t >= rt[1] and t < rt[2]:
        cv.putText(image, "Extiende el codo", (235, 470),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
    return image


def calibracion(holistic):
    global rt, rom, capture, mp_drawing, mp_holistic, puntos
    r = True
    while r == True:
        frame = get_image(capture)
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        result = holistic.process(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        try:
            lm_p = result.pose_landmarks
            if arm == 1:
                lm = result.right_hand_landmarks
            else:
                lm = result.left_hand_landmarks

            if lm_p is not None:
                coor = coor_obtain(lm_p.landmark, np.linspace(
                    1, 33, 33).astype(int).tolist(), puntos)
                if lm is not None:
                    image, coor[0, :] = reemplazar_wrist(
                        mp_drawing, mp_holistic, image, lm, [coor[0, 0], coor[0, 1]])
                for i in range(len(coor)-2):
                    cv.line(image, (int(coor[i, 0]), int(coor[i, 1])), (int(
                        coor[i+1, 0]), int(coor[i+1, 1])), (219, 230, 101), 2)
                for i in range(len(coor)-1):
                    cv.circle(image, (int(coor[i, 0]), int(
                        coor[i, 1])), 3, (102, 31, 208), 2)

                # Si cuadro es true, las coordenadas están bien posicionadas
                cuadro, image = encuadrar(coor, image)
                t_verde = medir_tiempo(cuadro)
                if t_verde >= rt[0]:
                    rom_calculate(coor)
                    if t_verde >= rt[2]:
                        r = False
                        print(rom)
                image = avisos(image, t_verde)
            else:
                image = colormarco(False, image)
        except:
            pass

        show_image(image)
        cv.waitKey(1) 

def cal():
    global mp_drawing, mp_holistic
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        calibracion(holistic)


def base_control():
    global Db, arm, user
    Db = Base.DataBase()
    user = Db.get_user()
    arm_b = Db.show_arm(user[1])
    print(arm_b)
    if arm_b[0] == 'R':
        arm = 0
    else:
        arm = 1


def write_base():
    Db.insert_data(maxang, minang, angtotal/j, maxv, maxw, vtotal/j, wtotal/j, rep, rep_sp, user[0])


def main():
    global arm, rom, rom_a, puntos, width, height, capture, mp_drawing, mp_holistic, rt, u,tlim,p_mano
    #base_control()
    ####
    arm=0
    tlim=30;
    ####
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    capture = cv.VideoCapture(0)
    rom = np.array([360, -360], dtype=float)
    u = 0
    width = 640
    height = 480
    #rt = [5, 15, 25]
    rt = [3, 7, 11]
    if arm == 0:
        puntos = [15, 13, 11, 12, 1]
    else:
        puntos = [16, 14, 12, 11, 1]
    p_mano=[2,17] #inicio de pulgar y de pinky
    cal()
    pyautogui.press("G") #para comenzar el juego
    rom_a = rom[1]-rom[0]
    print(rom_a)
    plt.figure("Angle transition")
    t1 = threading.Thread(target=Imagen, name="t1")
    t1.start()
    ani = FuncAnimation(plt.gcf(), animate, interval=700)
    plt.tight_layout()
    plt.show()
    t1.join()


if __name__ == "__main__":
    main()
