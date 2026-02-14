from ultralytics import YOLO

def main():
    model = YOLO(r"D:\Sem6_Subjects\Capstone\train7\weights\best.pt")   

    model.train(
        data="data.yaml",
        imgsz=1280,
        batch=4,
        epochs=30,
        patience=5,
        workers=4,
        project="D:/Sem6_Subjects/Capstone",
        device=0
    )

if __name__ == "__main__":
    main()
