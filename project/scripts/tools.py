import torch
import time
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class NumberPlateDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_ids = df['filename'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['filename'] == image_id]

        img_path = records.iloc[0]['image_path']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        boxes = []
        labels = []

        for _, row in records.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([hash(image_id) % (2**31)]),
            'filename': image_id
        }

        return image, target

def compute_iou(boxA, boxB):
    """
    Each box is defined as [x_min, y_min, x_max, y_max].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return iou

def calculate_image_iou(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Matches predicted boxes to ground truth boxes and calculates IoU.

    Returns:
        matched_ious: List of IoUs for matched boxes
        num_correct: Number of predictions with IoU > threshold
    """
    matched_ious = []
    num_correct = 0
    matched_gt = set()

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= iou_threshold:
            matched_ious.append(best_iou)
            num_correct += 1
            matched_gt.add(best_gt_idx)

    return matched_ious, num_correct

def train_one_epoch(model, data_loader, optimizer, device, epoch, print_freq=50):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for img in images:
            print(img.shape)  # (C, H, W)
            break

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        if (i + 1) % print_freq == 0:
            print(f"[Epoch {epoch} | Step {i + 1}] Loss: {total_loss.item():.4f}")

    avg_loss = epoch_loss / len(data_loader)
    print(f"\nEpoch {epoch} completed in {time.time() - start_time:.2f}s - Avg Loss: {avg_loss:.4f}")
    return avg_loss

        

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    total_iou = 0.0
    total_gt_boxes = 0
    matched_gt_boxes = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]


            outputs = model(images)

            for pred, gt in zip(outputs, targets):
                pred_boxes = pred['boxes'].cpu()
                gt_boxes = gt['boxes'].cpu()

                total_gt_boxes += len(gt_boxes)

                for gt_box in gt_boxes:
                    best_iou = 0.0
                    for pred_box in pred_boxes:
                        iou = compute_iou(gt_box.numpy(), pred_box.numpy())
                        best_iou = max(best_iou, iou)

                    total_iou += best_iou
                    if best_iou >= iou_threshold:
                        matched_gt_boxes += 1

    avg_iou = total_iou / total_gt_boxes
    detection_rate = matched_gt_boxes / total_gt_boxes
    print(f"\n🔍 Evaluation Results:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Detection Rate @ IoU > {iou_threshold}: {detection_rate * 100:.2f}%")

    return avg_iou, detection_rate