from __future__ import print_function
from datetime import date, datetime, timedelta
import mysql.connector

#establishing the connection
conn = mysql.connector.connect(
   user='root', password='maestro', host='127.0.0.1', database='maestro'
)

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

tomorrow = datetime.now().date() + timedelta(days=1)

add_employee = ("INSERT INTO employees "
               "(first_name, last_name, hire_date, gender, birth_date) "
               "VALUES (%s, %s, %s, %s, %s)")


data_employee = ('Geert', 'Vanderkelen', tomorrow, 'M', date(1977, 6, 14))

# Insert new employee
cursor.execute(add_employee, data_employee)
emp_no = cursor.lastrowid

# Make sure data is committed to the database
conn.commit()

cursor.close()
conn.close()