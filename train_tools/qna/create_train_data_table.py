import sys
path = "/Users/giyeon/Section4//Project4/project4_chatbot"
sys.path.append(path)

import pymysql
from config.DatabaseConfig import *

db = None
try:
    db = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        passwd = DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
        )

#테이블 구성
    
    sql = '''
      CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
      `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
      `intent` VARCHAR(45) NULL,
      `ner` VARCHAR(1024) NULL,
      `query` TEXT NULL,
      `answer` TEXT NOT NULL,
      PRIMARY KEY (`id`))
    ENGINE = InnoDB DEFAULT CHARSET=utf8
    '''

    with db.cursor() as cursor:
        cursor.execute(sql)

#에러 발생시 에러 출력

except Exception as e:
    print(e)

#데이터베이스가 만들어지면 cursor 닫기

finally:
    if db is not None:
        db.close()