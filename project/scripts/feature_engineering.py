from load_data import load_annotations
from database_connection import get_connection
import os
import mlflow


IMAGE_SIZE = 224

def create_plates_advanced_features(df):
    df['bbox_width']  = df['xmax'] - df['xmin']
    df['bbox_height'] = df['ymax'] - df['ymin']
    df['bbox_area']   = df['bbox_width'] * df['bbox_height']

    df['aspect_ratio'] = df['bbox_width'] / df['bbox_height']
    df['area_fraction'] = df['bbox_area'] / (df['width'] * df['height'])

    df['center_x_norm'] = ((df['xmin'] + df['xmax']) / 2) / df['width']
    df['center_y_norm'] = ((df['ymin'] + df['ymax']) / 2) / df['height']

    df['margin_left']   = df['xmin']
    df['margin_top']    = df['ymin']
    df['margin_right']  = df['width']  - df['xmax']
    df['margin_bottom'] = df['height'] - df['ymax']

    return df

def create_detection_image_features(df):
    df['xmin'] = df['xmin'] / (df['width'] / IMAGE_SIZE)
    df['xmax'] = df['xmax'] / (df['width'] / IMAGE_SIZE)
    df['ymin'] = df['ymin'] / (df['height'] / IMAGE_SIZE)
    df['ymax'] = df['ymax'] / (df['height'] / IMAGE_SIZE)
    return df

def save_plates_features_to_db(df):
    conn = get_connection()

    selected_cols = [
        'filename', 'bbox_width', 'bbox_height', 'bbox_area',
        'aspect_ratio', 'area_fraction', 'center_x_norm', 'center_y_norm',
        'margin_left', 'margin_top', 'margin_right', 'margin_bottom',
        'ocr_text', 'worth_ocr'
    ]

    df[selected_cols].to_sql(
        name='engineered_plate_features',
        con=conn,
        if_exists='replace',
        index=False
    )

    conn.commit()
    conn.close()
    print("✅ Engineered plate features saved to 'engineered_plate_features' in the database.")

def save_detection_features_to_db(df):
    conn = get_connection()
    df[['filename', 'xmin', 'xmax', 'ymin', 'ymax']].to_sql(
        name='engineered_detection_features',
        con=conn,
        if_exists='replace',
        index=False
    )
    conn.commit()
    conn.close()
    print("✅ Detection features saved to 'engineered_detection_features' in the database.")

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename}")

def feature_engineering_pipeline(row_limit=None):
    mlflow.set_experiment("feature_engineering")
    
    with mlflow.start_run(run_name="plate_feature_engineering"):
        print("📥 Loading annotation data...")
        df = load_annotations(row_limit=row_limit)

        mlflow.log_param("row_limit", row_limit if row_limit else "all")
        mlflow.log_metric("raw_rows", len(df))

        print("🔧 Creating detection features...")
        detection_df = create_detection_image_features(df)
        save_detection_features_to_db(detection_df)
        detection_path = os.path.join('content', 'detection_features.csv')
        save_to_csv(detection_df, detection_path)

        mlflow.log_metric("detection_rows", len(detection_df))
        mlflow.log_artifact(detection_path)

        print("🔧 Creating advanced plate features...")
        plate_df = create_plates_advanced_features(df)
        save_plates_features_to_db(plate_df)
        plate_path = os.path.join('content', 'engineered_plate_features.csv')
        save_to_csv(plate_df, plate_path)

        mlflow.log_metric("plate_rows", len(plate_df))
        mlflow.log_artifact(plate_path)

        mlflow.log_text(plate_df.head().to_string(), "sample_plate_features.txt")


if __name__ == "__main__":
    print("🚀 Starting feature engineering pipeline...")
    feature_engineering_pipeline()
    print("✅ Feature engineering pipeline complete.")
