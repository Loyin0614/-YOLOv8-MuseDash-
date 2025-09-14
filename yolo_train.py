from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")  # 加载预训练模型
    results = model.train(
        data="peropero.yaml",  # 使用自定义数据集进行训练
        epochs=500,             # 训练 200 个周期
        imgsz=640,             # 图像大小
        batch=16,              # 批量大小
        device='0'             # 使用 GPU 设备编号
    )
    return results

if __name__ == '__main__':
    train_model()