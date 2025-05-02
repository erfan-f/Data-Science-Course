import sqlite3
import pandas as pd

df = pd.read_excel('content/car_plate_annotations.xlsx')
df['image_path'] = 'content/images/' + df['filename']

conn = sqlite3.connect('database/dataset.db')

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