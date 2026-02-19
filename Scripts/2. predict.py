"""
Predict Indonesian Sign Language (SIBI) Alphabet using trained YOLOv8 model.

Supports prediction from image file, folder of images, or webcam.

Usage:
    python predict.py --source image.jpg
    python predict.py --source path/to/images/
    python predict.py --webcam
    python predict.py --source image.jpg --save
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Project root directory
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_MODEL = PROJECT_DIR / "Model" / "best.onnx"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SIBI Alphabet Detection using YOLOv8"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to image file or folder of images",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam for real-time detection",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help=f"Path to model file (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated results to output/ folder",
    )
    return parser.parse_args()


def load_model(model_path):
    """Load YOLO model."""
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Make sure to train the model first or place model files in Model/")
        sys.exit(1)

    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path, task="detect")
    return model


def predict_image(model, image_path, conf=0.5, save=False):
    """Run prediction on a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    results = model.predict(source=img, conf=conf, save=False)

    if results:
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                print(f"   No gesture detected in: {Path(image_path).name}")
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Draw bounding box
                color = (0, 200, 0)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = f"{class_name}: {confidence:.0%}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                img = cv2.rectangle(
                    img,
                    (x1, y1 - label_h - baseline - 5),
                    (x1 + label_w, y1),
                    color,
                    -1,
                )
                img = cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                print(f"   ✅ Detected: {class_name} ({confidence:.0%})")

    # Save or display
    if save:
        output_dir = PROJECT_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"pred_{Path(image_path).name}"
        cv2.imwrite(str(output_path), img)
        print(f"   💾 Saved: {output_path}")
    else:
        cv2.imshow("SIBI Detection", img)
        print("   Press any key to continue, 'q' to quit...")
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)


def predict_webcam(model, conf=0.5):
    """Run real-time prediction using webcam."""
    print("📷 Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, save=False, verbose=False)

        if results:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    # Draw bounding box
                    color = (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{class_name}: {confidence:.0%}"
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_h - baseline - 5),
                        (x1 + label_w, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

        cv2.imshow("SIBI Alphabet Detection (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📷 Webcam stopped.")


def main():
    args = parse_args()

    if not args.source and not args.webcam:
        print("❌ Please specify --source <path> or --webcam")
        print("   Example: python predict.py --source image.jpg")
        print("   Example: python predict.py --webcam")
        sys.exit(1)

    # Load model
    model = load_model(args.model)

    if args.webcam:
        predict_webcam(model, conf=args.conf)
    else:
        source = Path(args.source)

        if source.is_file():
            # Single image
            print(f"\n🔍 Predicting: {source.name}")
            predict_image(model, source, conf=args.conf, save=args.save)

        elif source.is_dir():
            # Folder of images
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            images = [
                f for f in source.iterdir()
                if f.suffix.lower() in image_exts
            ]

            if not images:
                print(f"❌ No images found in: {source}")
                sys.exit(1)

            print(f"\n🔍 Predicting {len(images)} images from: {source}")
            for img_path in sorted(images):
                print(f"\n📄 {img_path.name}")
                predict_image(model, img_path, conf=args.conf, save=args.save)

        else:
            print(f"❌ Source not found: {source}")
            sys.exit(1)

    cv2.destroyAllWindows()
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
