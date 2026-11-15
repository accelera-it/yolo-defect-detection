# Yolo defect detection

## Potato &amp; Carrot Defect Detection with YOLOv11

Run everything from the repository root.

```bash
pip install -r requirements.txt

streamlit run ui.py        # upload an image and see the detections (or: sh demo/run.sh)
python generate.py         # regenerate the synthetic dataset in dataset/ (replaces the images and labels)
python demo/train.py       # train on dataset/data.yaml; results go to runs/detect/
```

The dataset has six classes (`dataset/data.yaml`): `healthy_potato`, `potato_rot`, `potato_spot`, `healthy_carrot`, `carrot_crack`, `carrot_rot`. The weights trained on it are in `runs/detect/train/weights/best.pt`; `ui.py` currently uses the stock `models/yolo11n.pt`.