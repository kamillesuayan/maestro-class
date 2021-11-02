from __future__ import print_function
from datetime import date, datetime, timedelta
import mysql.connector

#establishing the connection
conn = mysql.connector.connect(
   user='root', password='maestro', host='127.0.0.1', database='maestro'
)

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

submission_time = datetime.now().date()

add_student_homework = ("INSERT INTO attack_homework "
               "(student_id, code_snippet, time) "
               "VALUES (%s, %s, %s)")

text_file = open("data.txt", "r")
 
#read whole file to a string
data = text_file.read()
 
#close file
text_file.close()
 

data_employee = (22766303, data, submission_time)

# Insert new employee
cursor.execute(add_student_homework, data_employee)
emp_no = cursor.lastrowid

# Make sure data is committed to the database
conn.commit()

cursor.close()
conn.close()