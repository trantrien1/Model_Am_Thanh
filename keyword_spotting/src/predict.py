import argparse
import json

import torch

from .artifacts import load_inference
from .features import LogMel, preprocess_audio


def predict(run_dir, audio_path, device="auto"):
    model, checkpoint, threshold_artifact, cfg, device = load_inference(run_dir, device)
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    audio, metadata = preprocess_audio(audio_path, checkpoint["audio_config"], reject_long=True)
    extractor = LogMel(checkpoint["feature_config"]).to(device)
    with torch.no_grad():
        features = extractor(torch.from_numpy(audio).unsqueeze(0).to(device))
        mean = torch.tensor(checkpoint["normalization"]["mean"], device=device).view(1, -1, 1)
        std = torch.tensor(checkpoint["normalization"]["std"], device=device).view(1, -1, 1)
        logits = model(((features - mean) / std).unsqueeze(1))
        probability = float(torch.sigmoid(logits).item())
    threshold = threshold_artifact["threshold"]
    return {"target_word": checkpoint["target_word"], "probability": probability,
            "threshold": threshold, "prediction": int(probability >= threshold),
            "duration_seconds": metadata["duration_seconds"]}


def main():
    parser = argparse.ArgumentParser(description="Predict keyword presence in a WAV up to 5 seconds; probabilities are uncalibrated scores.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        result = predict(args.run_dir, args.audio, args.device)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
