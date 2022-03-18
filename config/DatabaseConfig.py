DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "1234"
DB_NAME = "policy_info"

#Making global Keywords in a function

def DatabaseConfig():
    global DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME