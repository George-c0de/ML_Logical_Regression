import psycopg2
from settings import DBNAME, PORT, PASSWORD, USER, HOST

connect = psycopg2.connect(dbname=DBNAME, host=HOST, user=USER, password=PASSWORD, port=PORT)
with connect:
    with connect.cursor() as cursor:
        cursor.execute("SELECT * FROM mytable")
        print(cursor.fetchall())

