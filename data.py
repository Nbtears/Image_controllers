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
            print(user)
            return user

        except:
            pass

    def insert_data(self,aa,ai,vm,cm,va,ca,rep,id):
        sql = "INSERT INTO game (angle_max,angle_min,vel_max,acc_max,vel_avg,acc_avg,rep,sesion) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)".format()
        
        try:
            self.cursor.execute(sql,(aa,ai,vm,cm,va,ca,rep,id))
        except: 
            pass

    def print_data(self,name,id):
        sql= "SELECT * FROM {} WHERE sesion = {}". format(name,id)
        try:
            self.cursor.execute(sql)
            info = self.cursor.fetchone()
            print(info)
        except:
            pass

    def close(self):
        self.connection.close()

data = DataBase()
user = data.get_user()
data.insert_data(user[0],10)
name = 'game'
data.print_data(name,user[0])
data.close()



