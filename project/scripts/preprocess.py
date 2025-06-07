import pandas as pd
import os
import cv2
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import MinMaxScaler
from database_connection import get_connection, get_cursor
from load_data import load_detection_images_joined_data, load_plates_joined_data
import matplotlib.pyplot as plt

IMAGE_SIZE = 224

def preprocess_detection_images(output_folder, row_limit=None):
    print("📥 Loading detection image data from database...")
    df = load_detection_images_joined_data(row_limit=row_limit)

    os.makedirs(output_folder, exist_ok=True)

    print(f"🔄 Starting detection image preprocessing ({len(df['filename'].unique())} total)...")
    processed, skipped = 0, 0

    seen_filenames = set()

    for idx, row in df.iterrows():
        img_path = row['image_path']
        filename = row['filename']

        if filename in seen_filenames:
            continue  
        seen_filenames.add(filename)

        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Could not read: {img_path}")
            skipped += 1
            continue

        resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        out_filename = f"{filename}" 
        save_path = os.path.join(output_folder, out_filename)

        cv2.imwrite(save_path, resized)

        df.at[idx, 'image_path'] = save_path
        processed += 1

    print(f"✅ Detection images processed: {processed}, skipped: {skipped}")

    print("💾 Updating 'engineered_detection_features' in DB...")
    conn = get_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute("ALTER TABLE engineered_detection_features ADD COLUMN image_path TEXT;")
    except:
        pass

    for _, row in df.iterrows():
        if pd.notnull(row['filename']):
            cursor.execute("""
                UPDATE engineered_detection_features
                SET image_path = ?
                WHERE filename = ?
            """, (row['image_path'], row['filename']))

    conn.commit()
    conn.close()
    print("✅ Detection image metadata saved.\n")
    return df


def preprocess_plate_images(output_folder, resize_method='mean', row_limit=None):
    print("📥 Loading plate data from database...")
    df = load_plates_joined_data(row_limit=row_limit)

    os.makedirs(output_folder, exist_ok=True)

    if resize_method == 'mean':
        mean_w = int(df['bbox_width'].mean())
        mean_h = int(df['bbox_height'].mean())
        print(df['bbox_height'])
        print(df['bbox_width'])
        target_size = (mean_w, mean_h)
        print(f"picture target size is:({mean_w},{mean_h}) ")
    processed, skipped = 0, 0
    print(f"🚗 Starting license plate preprocessing ({len(df)} total)...")

    for idx, row in df.iterrows():
        img_path = row['image_path']
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Could not read: {img_path}")
            skipped += 1
            continue

        x0, y0, x1, y1 = row[['xmin', 'ymin', 'xmax', 'ymax']]
        cropped = image[y0:y1, x0:x1]
        resized = cv2.resize(cropped, target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm = (gray.astype('float32') / 255.0 * 255).astype('uint8')

        plate_name = f'plate_{idx:04d}.png'
        plate_path = os.path.join(output_folder, plate_name)
        # plt.imshow(norm)
        # plt.axis("off")
        # plt.show()

        cv2.imwrite(plate_path, norm)

        df.at[idx, 'preprocessed_plate'] = plate_name
        df.at[idx, 'plate_path'] = plate_path
        processed += 1

    print(f"✅ Plates processed: {processed}, skipped: {skipped}")

    print("💾 Updating 'engineered_plate_features' in DB...")
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
    print("✅ Plate image metadata saved.\n")
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
    print("🗑️ Removed empty bounding box annotations.\n")

def normalize_features():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT p.filename, f.bbox_width, f.bbox_height, f.bbox_area,
               f.aspect_ratio, f.area_fraction,
               f.center_x_norm, f.center_y_norm
        FROM image_annotations AS p
        JOIN engineered_plate_features AS f ON p.filename = f.filename
    """, conn)
    conn.close()

    scaler = MinMaxScaler()
    scale_cols = ['bbox_width', 'bbox_height', 'bbox_area',
                  'aspect_ratio', 'area_fraction', 'center_x_norm', 'center_y_norm']

    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

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
    print("📐 Features normalized.\n")

def drop_redundant_features():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM engineered_plate_features", conn)
    
    to_drop = ['bbox_area', 'margin_left', 'margin_right', 'margin_bottom']
    df_reduced = df.drop(columns=to_drop, errors='ignore')

    df_reduced.to_sql("engineered_plate_features", conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    print(f"❌ Dropped redundant columns: {to_drop}\n")

def detect_blurry_plates(plate_dir=os.path.join('content', 'plates'), threshold=100.0):
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
    print("🔍 Blur analysis complete.\n")

if __name__ == "__main__":
    print("🚀 Starting full preprocessing pipeline...\n")

    mlflow.set_experiment("preprocessing_pipeline")
    with mlflow.start_run(run_name="preprocess_v1"):
        plate_output_dir = os.path.join('content', 'plates')
        detection_output_dir = os.path.join('content', 'detection_images')
        blur_threshold = 100.0
        resize_method = 'mean'

        mlflow.log_param("resize_method", resize_method)
        mlflow.log_param("blur_threshold", blur_threshold)

        plate_df = preprocess_plate_images(plate_output_dir, resize_method=resize_method)
        mlflow.log_metric("plates_processed", len(plate_df))

        detection_df = preprocess_detection_images(detection_output_dir)
        mlflow.log_metric("detections_processed", len(detection_df))

        remove_invalid_annotations()
        normalize_features()
        drop_redundant_features()
        
        detect_blurry_plates(plate_output_dir, threshold=blur_threshold)

        final_df = pd.read_sql("SELECT * FROM engineered_plate_features", get_connection())
        final_path = os.path.join("content", "final_plate_metadata.csv")
        final_df.to_csv(final_path, index=False)
        mlflow.log_artifact(final_path)

        mlflow.log_text(final_df.head().to_string(), "plate_metadata_preview.txt")

    print("✅ All preprocessing complete.")
