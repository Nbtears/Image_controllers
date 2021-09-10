import cv2 as cv
import mediapipe as mp
import numpy as np

rep = 0
def angle_calculate(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle=360-angle
    
    return angle 

def image_process (frame,mp_drawing,mp_holistic,holistic):  
    global angle
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
        
        angle = angle_calculate(shoulder_L,elbow_L,wrist_L)
        #look angle
        cv.putText(image,str(angle),
                   tuple(np.multiply(elbow_L,[647,510]).astype(int)),
                         cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv.LINE_AA)
        game_controller(angle)
          
    except:
        pass
     #dibujar las articulaciones del cuerpo en la imagen
    mp_drawing.draw_landmarks(image, result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (102,31,208),thickness = 2,circle_radius = 3),
                              mp_drawing.DrawingSpec(color = (103,249,237),thickness = 2,circle_radius = 2))
    
    return image

def game_controller(angle):
    global stage, rep
    if angle > 110:
        stage = 0
    if angle < 60 and stage==0:
        print("hi")
        stage = 1
        rep +=1
        print(rep)
     
       
def main():
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
            image= image_process(frame,mp_drawing,mp_holistic,holistic)
            
            cv.imshow('camera',image)
            
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                capture.release() 
                break 
      
if __name__=="__main__":
    main()          