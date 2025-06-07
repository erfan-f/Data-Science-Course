import pandas as pd
from database_connection import get_connection

def load_annotations(row_limit=None):
    conn = get_connection()
    query = "SELECT * FROM image_annotations"
    if row_limit:
        query += f"LIMIT {row_limit}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_plate_features(row_limit=None):
    conn = get_connection()
    query = "SELECT * FROM engineered_plate_features"
    if row_limit:
        query += f" LIMIT {row_limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_detection_image_features(mode="train", row_limit=None):
    conn = get_connection()

    if mode == "train":
        query = """
            SELECT * FROM engineered_detection_features
            WHERE xmin IS NOT NULL AND ymin IS NOT NULL
              AND xmax IS NOT NULL AND ymax IS NOT NULL
        """
    elif mode == "predict":
        query = """
            SELECT * FROM engineered_detection_features
            WHERE xmin IS NULL OR ymin IS NULL
               OR xmax IS NULL OR ymax IS NULL
        """
    else:
        raise ValueError("mode must be 'train' or 'predict'")

    if row_limit:
        query += f" LIMIT {row_limit}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_plates_joined_data(row_limit=None):
    conn = get_connection()
    query = """
    SELECT
        a.filename,
        a.folder,
        a.width, a.height,
        a.xmin, a.ymin, a.xmax, a.ymax,
        a.image_path,
        f.bbox_width, f.bbox_height, f.bbox_area,
        f.aspect_ratio, f.area_fraction,
        f.center_x_norm, f.center_y_norm,
        f.margin_left, f.margin_top, f.margin_right, f.margin_bottom
    FROM image_annotations AS a
    JOIN engineered_plate_features AS f
      ON a.filename = f.filename
    """
    if row_limit:
        query += f" LIMIT {row_limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_detection_images_joined_data(row_limit=None):
    conn = get_connection()
    query = """
    SELECT
        a.filename,
        a.folder,
        a.width, a.height,
        a.xmin, a.ymin, a.xmax, a.ymax,
        a.image_path
    FROM image_annotations AS a
    JOIN engineered_detection_features AS f
      ON a.filename = f.filename
    """
    if row_limit:
        query += f" LIMIT {row_limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df