import pymysql
import openpyxl

import sys
path = "/Users/giyeon/Section4//Project4/project4_chatbot"
sys.path.append(path)

from config.DatabaseConfig import *

#db 초기화 함수

def all_clear_train_data(db):
    sql = """
    DELETE FROM chatbot_train_data
    """
    with db.cursor() as cursor:
        cursor.execute(sql)

#db에 excel 데이터 저장 함수

def insert_data(db, xls_row):
    intent, ner, query, answer = xls_row

    sql = """
    INSERT chatbot_train_data(intent, ner, query, answer)
    VALUES ('%s','%s','%s','%s') """ % (intent.value, ner.value, query.value, answer.value)

#엑셀에 데이터가 없는 경우 null로 저장
    sql = sql.replace("'None'", "null")

    with db.cursor() as cursor:
        cursor.execute(sql)
        print('{} 저장'.format(query.value))
        db.commit()

train_file = './train_data.xlsx'
db = None
try:
    db = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )

    all_clear_train_data(db)

    wb = openpyxl.load_workbook(train_file)
    sheet = wb['Sheet1']
    for row in sheet.iter_rows(min_row=2): #헤더는 저장하지 않음
        insert_data(db, row)

    wb.close()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()