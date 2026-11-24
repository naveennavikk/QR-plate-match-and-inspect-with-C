"""Evaluate a pair of classifier weights with the same decision logic as api.py.

Usage: python training/evaluate_rules.py <models_dir> [label]
Reports, for the test split and the held-out accept pairs, how often the
API would approve (QRStatus 1), reject (0) or send to manual review (null).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
TEST = HERE / "data" / "prepared" / "test"
HOLDOUT = HERE / "data" / "holdout_accept"
ALIASES = {"rescan properly scanify image": "rescan scanify image"}


def norm(label):
    label = " ".join(label.strip().split()).casefold()
    return ALIASES.get(label, label)


def decide(models, path):
    preds = []
    for m in models:
        out = m.predict(source=str(path), device="cpu", verbose=False)[0]
        p = out.probs.data.cpu().numpy().astype(float)
        preds.append({norm(out.names[i]): float(p[i]) for i in range(len(p))})
    classes = set().union(*preds)
    avg = {c: float(np.mean([s.get(c, 0.0) for s in preds])) for c in classes}
    top = max(avg, key=avg.get)
    winners = [max(s, key=s.get) for s in preds]
    if top == "accept+" and "nails not proper" in winners:
        return "manual", top
    if top == "stick plate properly" or any(w in top for w in ("shop", "store", "merchant", "metal")):
        return "manual", top
    return ("accept" if top == "accept+" else "reject"), top


def main():
    models_dir = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else models_dir.name
    models = [YOLO(str(p)) for p in sorted(models_dir.glob("*_cls_best.pt"))]
    report = {"models": label, "test": {}, "holdout_accept": {}}

    for cls_dir in sorted(d for d in TEST.iterdir() if d.is_dir()):
        outcomes, exact = Counter(), 0
        for img in cls_dir.glob("*.jpg"):
            decision, top = decide(models, img)
            outcomes[decision] += 1
            exact += top == norm(cls_dir.name)
        n = sum(outcomes.values())
        report["test"][cls_dir.name] = {"n": n, **outcomes, "exact_class": exact}

    for kind in ("plate", "house"):
        outcomes, reasons = Counter(), defaultdict(int)
        for img in sorted((HOLDOUT / kind).glob("*.jpg")):
            decision, top = decide(models, img)
            outcomes[decision] += 1
            if decision != "accept":
                reasons[top] += 1
        report["holdout_accept"][kind] = {
            "n": sum(outcomes.values()), **outcomes,
            "non_accept_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        }

    t = report["test"]
    acc = t.get("Accept+", {})
    rej = [v for k, v in t.items() if k != "Accept+"]
    rej_n = sum(v["n"] for v in rej)
    report["summary"] = {
        "test_accept_approved_pct": round(100 * acc.get("accept", 0) / max(1, acc.get("n", 0)), 1),
        "test_reject_caught_pct": round(100 * sum(v.get("reject", 0) + v.get("manual", 0) for v in rej) / max(1, rej_n), 1),
        "test_reject_wrongly_approved": sum(v.get("accept", 0) for v in rej),
        "test_reject_exact_rule_pct": round(100 * sum(v["exact_class"] for v in rej) / max(1, rej_n), 1),
        "holdout_plate_approved_pct": round(100 * report["holdout_accept"]["plate"].get("accept", 0)
                                            / max(1, report["holdout_accept"]["plate"]["n"]), 1),
    }
    out = HERE / "reports" / f"eval_{label}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
