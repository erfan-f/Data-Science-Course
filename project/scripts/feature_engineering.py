from load_data import load_annotations
from database_connection import get_connection, get_cursor

def create_advanced_features(df):

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

def save_features_to_db(df):

    conn = get_connection()

    df[['filename', 'bbox_width', 'bbox_height', 'bbox_area', 'aspect_ratio', 
        'area_fraction', 'center_x_norm', 'center_y_norm', 'margin_left', 
        'margin_top', 'margin_right', 'margin_bottom']].to_sql(
        name='engineered_plate_features',
        con=conn,
        if_exists='replace', 
        index=False
    )

    conn.commit()
    conn.close()
    print("✅ Engineered features saved to 'engineered_plate_features' in the database.")


def save_to_csv(df, filename='content/engineered_features.csv'):
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename}")

def feature_engineering_pipeline(row_limit=None):
    print("📥 Loading data...")
    df = load_annotations(row_limit=row_limit)

    print("🔧 Creating advanced features...")
    df = create_advanced_features(df)

    save_features_to_db(df)

    save_to_csv(df, 'content/engineered_features.csv')

if __name__ == "__main__":
    print("🚀 Starting feature engineering pipeline...")
    feature_engineering_pipeline()
    print("✅ Feature engineering pipeline complete.")