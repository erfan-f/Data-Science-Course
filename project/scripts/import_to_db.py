import sqlite3
import pandas as pd
import os


excel_path = os.path.join('content', 'car_plate_annotations.xlsx')

df = pd.read_excel(excel_path)

df['image_path'] = df['filename'].apply(lambda x: os.path.join('content', 'images', x))




db_path = os.path.join('database', 'dataset.db')

conn = sqlite3.connect(db_path)

df.to_sql(
    name='image_annotations',
    con=conn,
    if_exists='replace',
    index=False,
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