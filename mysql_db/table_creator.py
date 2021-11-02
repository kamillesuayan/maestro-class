import mysql.connector

#establishing the connection
conn = mysql.connector.connect(
   user='root', password='maestro', host='127.0.0.1', database='maestro'
)

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Dropping EMPLOYEE table if already exists.
cursor.execute("DROP TABLE IF EXISTS HOMEWORK")

#Creating table as per requirement
sql ='''CREATE TABLE attack_homework(
   student_id INT NOT NULL,
   code_snippet LONGBLOB NOT NULL,
   submission_time date NOT NULL,
   PRIMARY KEY ( student_id )
)'''
cursor.execute(sql)



#Closing the connection
conn.close()