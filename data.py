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
        sql = 'SELECT user FROM sesion ORDER BY date DESC LIMIT 1'.format()
        try :
            self.cursor.execute(sql)
            user = self.cursor.fetchone()

            print (user)

        except:
            pass

data = DataBase()
data.get_user()

