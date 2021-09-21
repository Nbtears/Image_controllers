import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import keyboard

def Regression_calculation(y_true,y_pred):
    x=np.array(y_pred).reshape((-1,1))
    y=np.array(y_true)
    model=LinearRegression()
    model.fit(x,y)
    print("Coeficiente de x: ", model.coef_)
    print("Intercepto: ", model.intercept_)
    

def angle_calculate(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle=360-angle
    
    return int(angle)  

def image_process (frame,mp_drawing,mp_holistic,holistic):  
    angle = 0
    #cambios de color y aplicar módulo holistic
    image= cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    result=holistic.process(image)
    image= cv.cvtColor(image,cv.COLOR_BGR2RGB)
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
        
        #coordenadas de brazo der
        shoulder_R = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow_R = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                  landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist_R = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
        
       
        angle = angle_calculate(shoulder_L,elbow_L,wrist_L)
        #look angle
        cv.putText(image,str(angle),
                   tuple(np.multiply(elbow_L,[647,510]).astype(int)),
                         cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv.LINE_AA)
        
    except:
        pass
     #dibujar las articulaciones del cuerpo en la imagen
    mp_drawing.draw_landmarks(image, result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (102,31,208),thickness = 2,circle_radius = 3),
                              mp_drawing.DrawingSpec(color = (103,249,237),thickness = 2,circle_radius = 2))
    return image,angle
    
    

def Mistake_calculation(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred,squared=False)
    print(y_pred)

    print("Error Cuadrático Medio", mse)

    plt.title("Root Mean Square Error")
    plt.plot(y_true,y_true,'r',label="Referencia")
    plt.plot(y_true, y_pred,'bo',label="Datos obtenidos")
    plt.ylabel('Predicted angle')
    plt.xlabel('Actual angle')
    plt.text(30,100,"RMSE = "+str(f"{mse:.2f}"),fontsize=18)
    plt.show  
    
def main(): 
    
    y_true = [30, 45, 60, 75, 90,105,120,135,150]
    y_pred = []
    i= 0 
    ft=4
    run = True
    #setup mediapie
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    
    #Abrir cámara web 
    capture = cv.VideoCapture(0) 
    with mp_holistic.Holistic(min_detection_confidence=0.8,min_tracking_confidence=0.8)as holistic:
        while run:
            
            #Lerr datos de camara web
            data,frame = capture.read()
            frame=cv.flip(frame,1)
            image,angle=image_process(frame,mp_drawing,mp_holistic,holistic)
            cv.circle(image,(100,80),60,(12,7,86),-1)
            cv.putText(image,str(y_true[i]),[55,95],cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2,cv.LINE_AA)
            cv.imshow('camera',image)
            
            
            if cv.waitKey(1) == ord('q'):
                break 
            
            if keyboard.is_pressed(" ") and ft>3:
                ft=0
                list.append(y_pred,angle)
                i+=1
                if i <= 8:
                    print("Registrado")
                else:
                    run = False
            
            ft+=1
    
    cv.destroyAllWindows()
    capture.release()  
    Mistake_calculation(y_true, y_pred)
    Regression_calculation(y_true, y_pred)

if __name__=="__main__":
    main()          