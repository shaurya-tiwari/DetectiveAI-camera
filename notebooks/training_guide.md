# Training YOLOv8 for Weapon Detection

This notebook guides you through training a custom YOLOv8 model to detect guns (and other weapons).

## 1. Prepare Dataset

You need a dataset of images with "gun" labels.
Format: YOLO format (txt files with `class_id x_center y_center width height`).

Structure:
```
datasets/
    weapons/
        data.yaml
        train/
            images/
            labels/
        val/
            images/
            labels/
```

**data.yaml content:**
```yaml
path: ../datasets/weapons  # dataset root dir
train: train/images
val: val/images

nc: 1  # number of classes
names: ['gun']  # class names
```

## 2. Install Dependencies

If running on Colab, run this first:
```python
!pip install ultralytics
```

## 3. Train the Model

Run the following code to start training.
**Note:** It is highly recommended to run this on a GPU (like Google Colab).

```python
from ultralytics import YOLO

# Load a pretrained model (recommended for transfer learning)
model = YOLO('yolov8n.pt') 

# Train the model
# data: path to your data.yaml
# epochs: number of training rounds (50-100 is usually good)
# imgsz: image size (640 is standard)
results = model.train(data='path/to/data.yaml', epochs=50, imgsz=640)
```

## 4. Export the Model

After training, the best model will be saved in `runs/detect/train/weights/best.pt`.

Copy this file to your project's `models/` folder:
```bash
cp runs/detect/train/weights/best.pt C:/Users/shaurya/Desktop/Code/DetectiveAI/models/best.pt
```

## 5. Verify

Run the following to check if your new model detects "gun":
```python
model = YOLO('runs/detect/train/weights/best.pt')
print(model.names)  # Should print {0: 'gun'}
```
