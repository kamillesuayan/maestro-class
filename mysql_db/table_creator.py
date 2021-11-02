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
sql ='''CREATE TABLE EMPLOYEE(
   ID CHAR(20) NOT NULL,
   LAST_NAME CHAR(20),
   AGE INT,
   SEX CHAR(1),
   INCOME FLOAT
)'''
cursor.execute(sql)

"""sql ='''CREATE TABLE ATTACK_PROJECT(
   FIRST_NAME CHAR(20) NOT NULL,
   LAST_NAME CHAR(20),
   AGE INT,
   SEX CHAR(1),
   INCOME FLOAT
)'''
cursor.execute(sql)


sql ='''CREATE TABLE DEFENSE_HOMEWORK(
   FIRST_NAME CHAR(20) NOT NULL,
   LAST_NAME CHAR(20),
   AGE INT,
   SEX CHAR(1),
   INCOME FLOAT
)'''
cursor.execute(sql)

sql ='''CREATE TABLE DEFENSE_PROJECT(
   FIRST_NAME CHAR(20) NOT NULL,
   LAST_NAME CHAR(20),
   AGE INT,
   SEX CHAR(1),
   INCOME FLOAT
)'''
cursor.execute(sql)"""


#Closing the connection
conn.close()