import pymysql

class DataBase():
    def __init__(self):
        self.connection = pymysql.connect(
           host = 'localhost', #ip
           user = 'root',
           password = '',
           db = 'save_the_axo' 
        )
        self.cursor = self.connection.cursor()

    def get_user(self):
        sql = 'SELECT id,user FROM sesion ORDER BY date DESC LIMIT 1'.format()
        try :
            self.cursor.execute(sql)
            user = self.cursor.fetchone()
            return user

        except:
            pass

    def show_arm(self,user):
        sql = "SELECT arm FROM user WHERE username = '{}'".format(user)
        try:
            self.cursor.execute(sql)
            arm = self.cursor.fetchone()
            print(arm)
            return arm
        except:
            pass

    def insert_data(self,aa,ai,vm,cm,va,ca,rep,id):
        sql = "INSERT INTO game (angle_max,angle_min,vel_max,acc_max,vel_avg,acc_avg,rep,sesion) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)".format()
        
        try:
            self.cursor.execute(sql,(aa,ai,vm,cm,va,ca,rep,id))
            self.connection.commit()
        except: 
            pass

    def print_data(self,name,key,id):
        # name tabla que queremos revisar
        # key identificador
        sql = "SELECT * FROM {} WHERE {} = '{}'".format(name,key,id)
        try:
            self.cursor.execute(sql,())
            info = self.cursor.fetchone()
            print(info)
        except:
            pass

    def add_calibration(self,a,b,name1,name2,id):
        # name1 y name2 datos regresion(m y b) 
        # o datos de umbrlaes del paciente(umax y umin)
        # a y b son esos datos
        sql = "INSERT INTO sesion ({},{}) VALUES (%s,%s) WHERE id = {}".format(name1,name2,id)
        try:
            self.cursor.execute(sql,(a,b))
            self.connection.commit()
        except:
            pass
    
    def show_calibration (self,name1,name2,id):
        sql = "SELECT {},{} FROM sesion WHERE id = {}".format(name1,name2,id)
        try:
            self.cursor.execute(sql)
            data = self.cursor.fetchone()
            return data
        except:
            pass
       
    def close(self):
        self.connection.close()
