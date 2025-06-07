import os
import mlflow
import mlflow.pytorch
import torch
from load_data import load_detection_image_features
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from tools import NumberPlateDataset, evaluate_model, train_one_epoch



def train_model(df, oTest = False):
    num_epochs = 10
    batch_size_train = 4
    batch_size_val = 2
    learning_rate = 0.005

    print("📊 Splitting dataset into train/val/test...")
    unique_filenames = df['filename'].unique()

    train_files, temp_files = train_test_split(unique_filenames, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    df_train = df[df['filename'].isin(train_files)].reset_index(drop=True)
    df_val = df[df['filename'].isin(val_files)].reset_index(drop=True)
    df_test = df[df['filename'].isin(test_files)].reset_index(drop=True)
    
    if oTest:
        return None, df_test

    print(f"✅ Dataset sizes — Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")


    train_dataset = NumberPlateDataset(df_train)
    val_dataset = NumberPlateDataset(df_val)

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn)


    print("📦 Initializing Faster R-CNN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🚀 Training on device: {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    ious = []
    detection_rates = []

    with mlflow.start_run(run_name="train_model"):
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size_train", batch_size_train)
        mlflow.log_param("batch_size_val", batch_size_val)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("device", str(device))

        print(f"📚 Starting training for {num_epochs} epochs...\n")
        for epoch in range(1, num_epochs + 1):
            print(f"📅 Epoch {epoch}/{num_epochs}")
            train_one_epoch(model, train_loader, optimizer, device, epoch)

            print(f"🔍 Evaluating model on validation set after epoch {epoch}...")
            avg_iou, detection_rate = evaluate_model(model, val_loader, device)
            print(f"📈 Avg IoU: {avg_iou:.4f} | Detection Rate: {detection_rate:.4f}\n")

            ious.append(avg_iou)
            detection_rates.append(detection_rate)

            mlflow.log_metric("avg_iou", avg_iou, step=epoch)
            mlflow.log_metric("detection_rate", detection_rate, step=epoch)
            
            lr_scheduler.step()
            print(f"📉 Learning rate stepped (epoch {epoch})\n")

        mlflow.pytorch.log_model(model, "fasterrcnn_model")

    print("✅ Training complete.")
    return model, df_test

def train_detection_model(oTest=False):
    print("📥 Loading features from database...")
    df = load_detection_image_features(mode='train')

    print("🏁 Starting training pipeline...")
    model, df_test = train_model(df, oTest=oTest)

    if oTest:
        return df_test

    save_dir = os.path.join('..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'detection_model.pth')
    torch.save(model.state_dict(), model_path)

    print(f"💾 Model saved to: {model_path}")

    mlflow.log_artifact(model_path)

    return df_test
