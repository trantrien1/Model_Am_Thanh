"""Score original recording regions with the same single-encoder CPU inference math."""
import torch
from frozen_phoneme import recording_features, aligned_window_features
from improve import TransferDetector


class RecordingPhoneDetector(TransferDetector):
    def __init__(self, model_path):
        super().__init__(model_path)
        if 'backbone' not in self.checkpoint or self.feature_context != 'recording':
            raise ValueError('Runtime evaluation requires packed recording-context weights.')

    def sequence(self, audio):
        with torch.inference_mode():
            return recording_features(self.acoustic, torch.from_numpy(audio), self.layers)

    def scores(self, sequence, starts):
        values = []
        with torch.inference_mode():
            for index in range(0, len(starts), 8):
                windows = aligned_window_features(sequence, starts[index:index + 8])
                probabilities = self.head(windows).softmax(-1)[:, :, self.ax]
                pool = self.checkpoint['pool_frames']
                if pool > 1:
                    probabilities = torch.nn.functional.avg_pool1d(probabilities.unsqueeze(1), pool, stride=1).squeeze(1)
                values.extend(probabilities.amax(-1).tolist())
        return values
