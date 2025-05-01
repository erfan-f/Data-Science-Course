import sqlite3
import pandas as pd
import os  


conn = sqlite3.connect('car_plates.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS car_plates (
    filename TEXT PRIMARY KEY,
    folder TEXT,
    width INTEGER,
    height INTEGER,
    xmin INTEGER,
    ymin INTEGER,
    xmax INTEGER,
    ymax INTEGER,
    image_path TEXT
)
''')
conn.commit()

df = pd.read_excel('car_plate_annotations.xlsx')

df['image_path'] = 'content/images/' + df['filename']  


df.to_sql(
    name='car_plates',
    con=conn,
    if_exists='replace',
    index=True,
    dtype={
        'filename': 'TEXT',
        'folder': 'TEXT',
        'width': 'INTEGER',
        'height': 'INTEGER',
        'xmin': 'INTEGER',
        'ymin': 'INTEGER',
        'xmax': 'INTEGER',
        'ymax': 'INTEGER',
        'image_path': 'TEXT'  
    }
)

conn.commit()
conn.close()
print("Database created successfully!")