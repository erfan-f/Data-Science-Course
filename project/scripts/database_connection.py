import sqlite3
import os

DEFAULT_DB_PATH = os.path.join('database', 'dataset.db')

def get_connection(db_path: str = DEFAULT_DB_PATH):

    if not os.path.exists(db_path):
        raise FileNotFoundError("DB NOT FOUND!")
    return sqlite3.connect(db_path)

def get_cursor(conn: sqlite3.Connection):
    return conn.cursor()