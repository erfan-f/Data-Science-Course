import sqlite3
import pandas as pd
import os

excel_path = os.path.join('content', 'car_plate_annotations.xlsx')
db_path = os.path.join('database', 'dataset.db')

df = pd.read_excel(excel_path)

df['filename'] = df['filename'].astype(str).str.strip()
df['ocr_text'] = df['ocr_text'].astype(str).str.strip()
df['image_path'] = df['filename'].apply(lambda x: os.path.join('content', 'images', x))

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
        'ocr_text': 'TEXT',
        'worth_ocr': 'INTEGER',
        'image_path': 'TEXT'
    }
)

conn.commit()
conn.close()

print("✅ Database created and annotation table inserted successfully!")
