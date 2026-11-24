"""Fine-tune the YOLO26n / YOLO11n classifiers on the accept/reject rule dataset.

Run prepare_rules_data.py first. Trained weights are written to
training/models/; copy them to ../models/ after reviewing evaluate_rules.py.
"""
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "prepared"
OUT_MODELS = HERE / "models"
RUNS = HERE / "runs"
PRETRAINED = Path.home() / "Documents" / "QR_House_Plate_AI_Inspector_Hybrid_v2"

EPOCHS = 40
IMAGE_SIZE = 224
BATCH = 16
PATIENCE = 12

MODEL_CONFIGS = [
    ("yolo26n-cls.pt", "yolo26n_cls"),
    ("yolo11n-cls.pt", "yolo11n_cls"),
]


def main():
    if not (DATA / "train").is_dir():
        raise SystemExit("Run prepare_rules_data.py first.")
    OUT_MODELS.mkdir(exist_ok=True)
    report = {"epochs": EPOCHS, "image_size": IMAGE_SIZE, "models": {}}
    for checkpoint, run_name in MODEL_CONFIGS:
        local = PRETRAINED / checkpoint
        started = time.time()
        dest = OUT_MODELS / f"{run_name}_best.pt"
        last = RUNS / run_name / "weights" / "last.pt"
        if dest.is_file():
            # Finished in an earlier session; only re-evaluate it.
            print(f"{run_name} already trained; skipping to evaluation.")
        elif last.is_file():
            # Interrupted (e.g. PC switched off): continue from the last saved epoch.
            print(f"Resuming {run_name} from {last}")
            YOLO(str(last)).train(resume=True)
        else:
            model = YOLO(str(local) if local.is_file() else checkpoint)
            model.train(
                data=str(DATA), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH,
                device="cpu", workers=0, patience=PATIENCE, project=str(RUNS),
                name=run_name, exist_ok=True, plots=True, amp=False, seed=42,
                fliplr=0.0,  # plates carry text; mirrored text is unrealistic
            )
        if not dest.is_file():
            shutil.copy2(RUNS / run_name / "weights" / "best.pt", dest)
        metrics = YOLO(str(dest)).val(data=str(DATA), split="test", imgsz=IMAGE_SIZE,
                                       batch=BATCH, device="cpu", workers=0, plots=False)
        report["models"][run_name] = {
            "weights": str(dest.relative_to(HERE)),
            "seconds": round(time.time() - started, 1),
            "test_top1": float(metrics.top1),
            "test_top5": float(metrics.top5),
        }
        print(f"Saved {dest}  test top1={metrics.top1:.3f}")
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "reports" / "training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
