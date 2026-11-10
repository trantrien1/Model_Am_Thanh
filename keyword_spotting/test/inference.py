"""Readable CPU/GPU inference using the frozen checkpoint and validation threshold."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from src.artifacts import load_inference
from src.features import LogMel, preprocess_audio

PROJECT = Path(__file__).resolve().parent


class KeywordSpotter:
    def __init__(self, run_dir=None, device="cpu"):
        self.run_dir = Path(run_dir or PROJECT / "model").resolve()
        missing = [name for name in ("best.pt", "threshold.json") if not (self.run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {', '.join(missing)} in {self.run_dir}. Keep best.pt and threshold.json together, or set --run-dir.")
        self.model, self.checkpoint, self.threshold_artifact, self.cfg, self.device = load_inference(self.run_dir, device)
        torch.set_num_threads(self.cfg["train"]["cpu_threads"])
        self.extractor = LogMel(self.checkpoint["feature_config"]).to(self.device).eval()
        stats = self.checkpoint["normalization"]
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.device).view(1, -1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32, device=self.device).view(1, -1, 1)

    def describe(self):
        return {"weights": str(self.run_dir / "best.pt"), "threshold_file": str(self.run_dir / "threshold.json"),
                "target_word": self.checkpoint["target_word"], "selected_epoch": self.checkpoint["epoch"],
                "threshold": self.threshold_artifact["threshold"], "device": str(self.device),
                "sample_rate": self.checkpoint["audio_config"]["sample_rate"],
                "maximum_duration_seconds": self.checkpoint["audio_config"]["duration_seconds"],
                "checkpoint_sha256": self.threshold_artifact["checkpoint_sha256"],
                "smoke_subset": self.checkpoint.get("smoke_subset", False)}

    @torch.inference_mode()
    def predict(self, audio_path):
        path = Path(audio_path).resolve()
        audio, metadata = preprocess_audio(path, self.checkpoint["audio_config"], reject_long=True)
        feature = self.extractor(torch.from_numpy(audio).unsqueeze(0).to(self.device))
        logits = self.model(((feature - self.mean) / self.std).unsqueeze(1))
        probability = float(torch.sigmoid(logits).item())
        threshold = self.threshold_artifact["threshold"]
        prediction = int(probability >= threshold)
        return {"audio": str(path), "target_word": self.checkpoint["target_word"], "probability": probability,
                "threshold": threshold, "prediction": prediction,
                "decision": "target_present" if prediction else "target_absent",
                "duration_seconds": metadata["duration_seconds"], "error": ""}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Infer keyword presence in WAV files up to 5 seconds. Weights and preprocessing come from the trained run.")
    parser.add_argument("--run-dir", default=str(PROJECT / "model"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", nargs="+", help="One or more WAV paths.")
    inputs.add_argument("--folder", help="Recursively predict all WAV files in a directory.")
    inputs.add_argument("--inspect", action="store_true", help="Show checkpoint path, epoch, target and threshold.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of readable results.")
    parser.add_argument("--csv", help="Save per-file predictions and any errors to CSV.")
    args = parser.parse_args(argv)
    try:
        spotter = KeywordSpotter(args.run_dir, args.device)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    description = spotter.describe()
    if args.inspect:
        print(json.dumps(description, indent=2, ensure_ascii=False))
        return 0
    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            parser.error(f"Audio folder does not exist: {folder}")
        paths = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    else:
        paths = [Path(path) for path in args.audio]
    if not paths:
        parser.error("No WAV files found.")
    if not args.json:
        print(f"Weights: {description['weights']}\nTarget: {description['target_word']} | Epoch: {description['selected_epoch']} | Device: {description['device']}")
        print("Quy tắc: score >= ngưỡng validation → phát hiện từ khóa. Score chưa được hiệu chuẩn thống kê.\n")
    results = []
    for path in paths:
        try:
            result = spotter.predict(path)
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            result = {"audio": str(path.resolve()), "target_word": description["target_word"], "probability": "", "threshold": description["threshold"],
                      "prediction": "", "decision": "error", "duration_seconds": "", "error": str(error)}
        results.append(result)
        if not args.json:
            print(result["audio"])
            if result["error"]:
                print("  Lỗi:", result["error"])
            else:
                decision = "PHÁT HIỆN" if result["prediction"] else "KHÔNG PHÁT HIỆN"
                print(f"  {decision} từ '{result['target_word']}' | score={result['probability']:.6f} | ngưỡng={result['threshold']:.2f} | dài={result['duration_seconds']:.3f}s")
    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        if not args.json:
            print("\nĐã lưu:", output.resolve())
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
    return int(any(result["error"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
