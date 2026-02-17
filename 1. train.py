"""
Train YOLOv8 model for Indonesian Sign Language (SIBI) Alphabet Detection.

Trains a YOLOv8 model on the USIBI dataset from Roboflow to detect
26 alphabet letters (A-Z) in Indonesian Sign Language.

Usage:
    python train.py
    python train.py --epochs 100 --model yolov8s.pt --imgsz 640
"""

import os
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv

from roboflow import Roboflow
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for SIBI Alphabet Detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pre-trained model variant (default: yolov8n.pt). "
             "Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Optimizer to use (default: Adam)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Export best model to ONNX format after training (default: True)",
    )
    return parser.parse_args()


def download_dataset():
    """Download USIBI dataset from Roboflow."""
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found. "
            "Create a .env file with: ROBOFLOW_API_KEY=your_key_here"
        )

    print("📥 Downloading USIBI dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("usibi-image-translate").project("usibi-jueew")
    version = project.version(2)
    dataset = version.download("yolov8")

    print(f"✅ Dataset downloaded to: {dataset.location}")
    return dataset


def train(args, dataset):
    """Train YOLOv8 model."""
    print(f"\n🚀 Starting training...")
    print(f"   Model     : {args.model}")
    print(f"   Epochs    : {args.epochs}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Optimizer : {args.optimizer}")

    model = YOLO(args.model)

    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        plots=True,
    )

    print("✅ Training complete!")
    return model, results


def export_and_save(model, export_onnx=True):
    """Export model to ONNX and save to model directory."""
    project_dir = Path(__file__).parent
    model_dir = project_dir / "Model"
    model_dir.mkdir(exist_ok=True)

    # Find the best model from training
    best_pt = Path("runs/detect/train/weights/best.pt")
    last_pt = Path("runs/detect/train/weights/last.pt")

    if best_pt.exists():
        dest = model_dir / "best.pt"
        shutil.copy2(best_pt, dest)
        print(f"📦 Saved best.pt → {dest}")

    if last_pt.exists():
        dest = model_dir / "last.pt"
        shutil.copy2(last_pt, dest)
        print(f"📦 Saved last.pt → {dest}")

    # Export to ONNX
    if export_onnx and best_pt.exists():
        print("📦 Exporting to ONNX format...")
        model_to_export = YOLO(str(best_pt))
        model_to_export.export(format="onnx")

        onnx_path = best_pt.with_suffix(".onnx")
        if onnx_path.exists():
            dest = model_dir / "best.onnx"
            shutil.copy2(onnx_path, dest)
            print(f"📦 Saved best.onnx → {dest}")

    # Copy training results
    result_dir = project_dir / "Result"
    result_dir.mkdir(exist_ok=True)

    results_csv = Path("runs/detect/train/results.csv")
    if results_csv.exists():
        shutil.copy2(results_csv, result_dir / "results.csv")

    # Copy plot images
    for plot_file in Path("runs/detect/train").glob("*.png"):
        shutil.copy2(plot_file, result_dir / plot_file.name)

    print(f"📊 Training results saved to: {result_dir}")


def main():
    args = parse_args()

    # Step 1: Download dataset
    dataset = download_dataset()

    # Step 2: Train model
    model, results = train(args, dataset)

    # Step 3: Export and save
    export_and_save(model, export_onnx=args.export_onnx)

    print("\n🎉 All done! Check the Model/ and Result/ folders for outputs.")


if __name__ == "__main__":
    main()
