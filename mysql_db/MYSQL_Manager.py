from __future__ import print_function
from datetime import date, datetime, timedelta
import mysql.connector


class MYSQL_Manager:
    def __init__(self):
        #establishing the connection
        self.conn = mysql.connector.connect(
        user='root', password='maestro', host='127.0.0.1', database='maestro'
        )
    def close_connection(self):
        self.conn.close()

    def input_attack_homework(self, student_id, data):
        #Creating a cursor object using the cursor() method
        cursor = self.conn.cursor()
        if data == None:
            text_file = open("data.txt", "r")
            #read whole file to a string
            data = text_file.read()
            #close file
            text_file.close()
        now = datetime.now()
        # dd/mm/YY H:M:S
        submission_time = now.strftime("%d/%m/%Y %H:%M:%S")
        add_student_homework = ("INSERT INTO attack_homework "
                    "(student_id, code_snippet, submission_time) "
                    "VALUES (%s, %s, %s)")

        data_student = (student_id, data, submission_time)
        # Insert new employee
        cursor.execute(add_student_homework, data_student)

        # Make sure data is committed to the database
        self.conn.commit()

        cursor.close()

    def input_attack_project(self, student_id, data):
        #Creating a cursor object using the cursor() method
        cursor = self.conn.cursor()
        if data == None:
            text_file = open("data.txt", "r")
            #read whole file to a string
            data = text_file.read()
            #close file
            text_file.close()
        now = datetime.now()
        # dd/mm/YY H:M:S
        submission_time = now.strftime("%d/%m/%Y %H:%M:%S")
        add_student_project = ("INSERT INTO attack_project "
                    "(student_id, code_snippet, submission_time) "
                    "VALUES (%s, %s, %s)")

        data_student = (student_id, data, submission_time)
        # Insert new employee
        cursor.execute(add_student_project, data_student)

        # Make sure data is committed to the database
        self.conn.commit()

        cursor.close()

    def input_defense_homework(self, student_id, data, model):
        #Creating a cursor object using the cursor() method
        cursor = self.conn.cursor()
        if data == None:
            text_file = open("defense.txt", "r")
            #read whole file to a string
            data = text_file.read()
            #close file
            text_file.close()
        now = datetime.now()
        # dd/mm/YY H:M:S
        submission_time = now.strftime("%d/%m/%Y %H:%M:%S")
        add_student_homework_df= ("INSERT INTO attack_project "
                    "(student_id, code_snippet, model, submission_time) "
                    "VALUES (%s, %s, %s, %s)")

        data_student = (student_id, data, model, submission_time)
        # Insert new employee
        cursor.execute(add_student_homework_df, data_student)

        # Make sure data is committed to the database
        self.conn.commit()

        cursor.close()
    def input_defense_project(self, student_id, data, model):
        #Creating a cursor object using the cursor() method
        cursor = self.conn.cursor()
        if data == None:
            text_file = open("defense.txt", "r")
            #read whole file to a string
            data = text_file.read()
            #close file
            text_file.close()
        now = datetime.now()
        # dd/mm/YY H:M:S
        submission_time = now.strftime("%d/%m/%Y %H:%M:%S")
        add_student_project_df= ("INSERT INTO attack_project "
                    "(student_id, code_snippet, model, submission_time) "
                    "VALUES (%s, %s, %s, %s)")

        data_student = (student_id, data, model, submission_time)
        # Insert new employee
        cursor.execute(add_student_project_df, data_student)

        # Make sure data is committed to the database
        self.conn.commit()

        cursor.close()

    def generate_tables(self):
        #Creating a cursor object using the cursor() method
        cursor = self.conn.cursor()

        #Dropping EMPLOYEE table if already exists.
        cursor.execute("DROP TABLE IF EXISTS HOMEWORK")

        #Creating table as per requirement
        sql ='''CREATE TABLE attack_homework(
        student_id INT NOT NULL,
        code_snippet LONGBLOB NOT NULL,
        submission_time date,
        grade INT,
        PRIMARY KEY ( student_id )
        )'''
        cursor.execute(sql)

        #Creating table as per requirement
        sql ='''CREATE TABLE attack_project(
        student_id INT NOT NULL,
        code_snippet LONGBLOB NOT NULL,
        submission_time date,
        grade INT,
        PRIMARY KEY ( student_id )
        )'''
        cursor.execute(sql)

        #Creating table as per requirement
        sql ='''CREATE TABLE defense_homework(
        student_id INT NOT NULL,
        code_snippet LONGBLOB NOT NULL,
        model LONGBLOB NOT NULL,
        submission_time date,
        grade INT,
        PRIMARY KEY ( student_id )
        )'''
        cursor.execute(sql)

        #Creating table as per requirement
        sql ='''CREATE TABLE defense_project(
        student_id INT NOT NULL,
        code_snippet LONGBLOB NOT NULL,
        model LONGBLOB NOT NULL,
        submission_time date,
        grade INT,
        PRIMARY KEY ( student_id )
        )'''
        cursor.execute(sql)
        cursor.close()


def main():
    manager = MYSQL_Manager()

    if True:
        manager.generate_tables()
    else:
        manager.input_attack_homework(22466303, None)

if __name__ == "__main__":
    main()