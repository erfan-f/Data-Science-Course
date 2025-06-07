import torch
import os
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from load_data import load_detection_image_features
from database_connection import get_connection
from sklearn.model_selection import train_test_split
from tools import evaluate_model, NumberPlateDataset



def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(model_path: str, device: torch.device):
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_on_test_set(df_test):
    print("📦 Creating test dataset...")
    test_dataset = NumberPlateDataset(df_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'detection_model.pth')
    model_path = os.path.abspath(model_path)
    model = load_model(model_path, device)
    model.to(device)

    print("📝 Writing predictions to the database...")
    conn = get_connection()
    cursor = conn.cursor()

    with torch.no_grad():
        for images, metas in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for meta, output in zip(metas, outputs):
                boxes = output['boxes'].cpu().numpy()
                if len(boxes) == 0:
                    continue
                xmin, ymin, xmax, ymax = boxes[0].astype(int)

                cursor.execute("""
                    UPDATE engineered_detection_features
                    SET xmin = ?, ymin = ?, xmax = ?, ymax = ?
                    WHERE filename = ?
                """, (xmin, ymin, xmax, ymax, meta["filename"]))

    conn.commit()
    conn.close()
    print("✅ Predictions saved to the database.")

def evaluation_on_test_set(df_test):
    print("📦 Creating test dataset...")
    test_dataset = NumberPlateDataset(df_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'detection_model.pth')
    model_path = os.path.abspath(model_path)
    model = load_model(model_path, device)
    model.to(device)

    with mlflow.start_run(run_name="detection_evaluation"):
        print("📈 Evaluating model on test set...")
        avg_iou, detection_rate = evaluate_model(model, test_loader, device)

        mlflow.log_metric("test_avg_iou", avg_iou)
        mlflow.log_metric("test_detection_rate", detection_rate)

        mlflow.log_param("device", str(device))
        mlflow.pytorch.log_model(model, "model")

        print(f"✅ Logged to MLflow: IoU = {avg_iou:.4f}, Detection Rate = {detection_rate:.4f}")



# if __name__ == "__main__":
#     df = load_detection_image_features(mode="test")
#     predict_on_test_set(df)
