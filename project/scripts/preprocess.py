import pandas as pd
import os
import cv2
from sklearn.preprocessing import MinMaxScaler
from database_connection import get_connection, get_cursor
from load_data import load_detection_images_joined_data, load_plates_joined_data


IMAGE_SIZE = 224

def preprocess_detection_images(output_folder, row_limit=None):
    print("📥 Loading data from database for car preprocessing...")
    df = load_detection_images_joined_data(row_limit=row_limit)

    print(f"📁 Creating output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    total = len(df)
    print(f"🔄 Starting detection image preprocessing for {total} entries...")
    processed = 0
    skipped = 0

    for idx, row in df.iterrows():
        img_path = row['image_path']
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Skipping unreadable detection image: {img_path}")
            skipped += 1
            continue

        image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image_name = f'image_{idx:04d}.png'
        image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(image_path, image_resized)
        df.at[idx, 'detection_image'] = image_name
        df.at[idx, 'image_path'] = image_path

        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"✅ Processed {processed}/{total} detection images...")

    print(f"✅ Finished detection image preprocessing: {processed} processed, {skipped} skipped.")

    print("💾 Updating database with preprocessed detection image filenames...")
    conn = get_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute("ALTER TABLE engineered_detection_features ADD COLUMN detection_image TEXT;")
        cursor.execute("ALTER TABLE engineered_detection_features ADD COLUMN image_path TEXT;")
    except:
        pass

    for _, row in df.iterrows():
        if pd.notnull(row['detection_image']):
            cursor.execute("""
                UPDATE engineered_detection_features
                SET detection_image = ?, image_path = ?
                WHERE filename = ?
            """, (row['detection_image'], row['image_path'], row['filename']))

    conn.commit()
    conn.close()
    print("✅ Plate data update complete.")

    return df


def preprocess_plate_images(output_folder, resize_method='mean', row_limit=None):

    print("📥 Loading data from database...")
    df = load_plates_joined_data(row_limit=row_limit)

    print(f"📁 Creating output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    df['preprocessed_plate'] = None

    if resize_method == 'mean':
        mean_w = int(df['bbox_width'].mean())
        mean_h = int(df['bbox_height'].mean())
        target_size = (mean_w, mean_h)

    total = len(df)
    processed = 0
    skipped = 0
    print(f"🚗 Starting preprocessing of {total} license plate images...")

    for idx, row in df.iterrows():
        img_path = row['image_path']
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Skipping unreadable image: {img_path}")
            skipped += 1
            continue

        x0, y0, x1, y1 = row[['xmin', 'ymin', 'xmax', 'ymax']]
        cropped = image[y0:y1, x0:x1]
        resized = cv2.resize(cropped, target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm = (gray.astype('float32') / 255.0 * 255).astype('uint8')

        plate_name = f'plate_{idx:04d}.png'
        plate_save_path = os.path.join(output_folder, plate_name)
        cv2.imwrite(plate_save_path, norm)
        df.at[idx, 'preprocessed_plate'] = plate_name
        df.at[idx, 'plate_path'] = plate_save_path

        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"✅ Processed {processed}/{total} license plate images...")

    print(f"✅ Finished license plate preprocessing: {processed} processed, {skipped} skipped.")

    print("💾 Updating database with preprocessed plate filenames...")
    conn = get_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute("ALTER TABLE engineered_plate_features ADD COLUMN preprocessed_plate TEXT;")
        cursor.execute("ALTER TABLE engineered_plate_features ADD COLUMN plate_path TEXT;")
    except:
        pass

    for _, row in df.iterrows():
        if pd.notnull(row['preprocessed_plate']):
            cursor.execute("""
                UPDATE engineered_plate_features
                SET preprocessed_plate = ?, plate_path = ?
                WHERE filename = ?
            """, (row['preprocessed_plate'], row['plate_path'], row['filename']))

    conn.commit()
    conn.close()
    print("✅ Plate data update complete.")

    return df


def remove_invalid_annotations():
    conn = get_connection()
    cursor = get_cursor(conn)
    cursor.execute("""
        DELETE FROM image_annotations
        WHERE xmin = 0 AND ymin = 0 AND xmax = 0 AND ymax = 0;
    """)
    conn.commit()
    conn.close()
    print("🗑️ Removed entries with no plate annotation.")


def normalize_features():
    conn = get_connection()
    query = """
    SELECT 
        p.filename, f.bbox_width, f.bbox_height,
        f.bbox_area, f.aspect_ratio, f.area_fraction,
        f.center_x_norm, f.center_y_norm
    FROM image_annotations AS p
    JOIN engineered_plate_features AS f
      ON p.filename = f.filename
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    cols_to_scale = ['bbox_width', 'bbox_height', 'bbox_area', 'aspect_ratio', 
                     'area_fraction', 'center_x_norm', 'center_y_norm']
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

    conn = get_connection()
    cursor = get_cursor(conn)
    for _, row in df_scaled.iterrows():
        cursor.execute("""
            UPDATE engineered_plate_features
            SET bbox_width = ?, bbox_height = ?, bbox_area = ?, aspect_ratio = ?,
                area_fraction = ?, center_x_norm = ?, center_y_norm = ?
            WHERE filename = ?
        """, (
            row['bbox_width'], row['bbox_height'], row['bbox_area'], row['aspect_ratio'],
            row['area_fraction'], row['center_x_norm'], row['center_y_norm'], row['filename']
        ))
    conn.commit()
    conn.close()
    print("📐 Normalized numeric features.")


def drop_redundant_features():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM engineered_plate_features", conn)
    
    to_drop = ['bbox_area', 'margin_left', 'margin_right', 'margin_bottom']
    df_reduced = df.drop(columns=to_drop, errors='ignore')

    df_reduced.to_sql(
        name='engineered_plate_features',
        con=conn,
        if_exists='replace',
        index=False
    )
    conn.commit()
    conn.close()
    print(f"❌ Dropped redundant features: {to_drop}")


def detect_blurry_plates(plate_dir= os.path.join('content', 'plates'), threshold=100.0):
    def is_blurry(image, threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap.var() < threshold, lap.var()

    results = []

    for fname in os.listdir(plate_dir):
        path = os.path.join(plate_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        blurry, score = is_blurry(img, threshold)
        results.append({
            'preprocessed_plate': fname,
            'blur_score': score,
            'is_blurry': int(blurry)
        })

    df_blur = pd.DataFrame(results)

    conn = get_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute("ALTER TABLE engineered_plate_features ADD COLUMN is_blurry INTEGER;")
        cursor.execute("ALTER TABLE engineered_plate_features ADD COLUMN blur_score REAL;")
    except:
        pass

    for _, row in df_blur.iterrows():
        cursor.execute("""
            UPDATE engineered_plate_features
            SET is_blurry = ?, blur_score = ?
            WHERE preprocessed_plate = ?
        """, (row['is_blurry'], row['blur_score'], row['preprocessed_plate']))

    conn.commit()
    conn.close()
    print("🔍 Blurry images analyzed and flagged in database.")


if __name__ == "__main__":
    print("🚀 Starting full data preprocessing pipeline...")
    preprocess_plate_images(os.path.join('content', 'plates'))
    preprocess_detection_images(os.path.join('content', 'detection_images'))
    remove_invalid_annotations()
    normalize_features()
    drop_redundant_features()
    detect_blurry_plates()
    print("✅ Preprocessing pipeline complete.\n")