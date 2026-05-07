"""
sentinel/acoustic/classifier.py

Real-time mosquito wingbeat classifier for Raspberry Pi Zero 2W.

THE HONEST FRAMING:
This is a research prototype. We classify on Aedes aegypti wingbeat patterns
because that's what's available open-source (HumBug / Abuzz datasets).
European invasive Aedes albopictus has a different but overlapping wingbeat
signature. Producing a deployment-grade European classifier requires labelled
field data — exactly what an LSHTM Logan Group / ECDC partnership would unlock.

For the demo, this is enough to show the sensing concept works in real time.
We say this on stage; we don't pretend otherwise.

DEPLOYMENT NOTES:
- Tested on Raspberry Pi Zero 2W with USB lavalier microphone.
- Uses TensorFlow Lite (smaller, faster than full TF on ARM).
- Classifies in 1-second windows; mosquito wingbeats are 300-600 Hz fundamental.

HARDWARE BILL OF MATERIALS (~£75-90):
- Raspberry Pi Zero 2W                     ~£18
- microSD 32GB                             ~£8
- USB lavalier microphone (Boya BY-LM10)   ~£15
- Small OLED screen (SSD1306, I2C)         ~£8
- Power supply 5V 2.5A                     ~£10
- Case, jumpers, mini speaker (optional)   ~£15

INSTALLATION (on the Pi):
    sudo apt update && sudo apt install -y python3-pip portaudio19-dev libatlas-base-dev
    pip3 install numpy sounddevice scipy tflite-runtime

RUNNING:
    python3 classifier.py
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

# These imports are guarded so the file can be inspected on a dev machine
# without portaudio / tflite installed.
try:
    import numpy as np
    import sounddevice as sd
    from scipy import signal as scipy_signal
except ImportError:
    np = None
    sd = None
    scipy_signal = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("classifier")

# ---------- Configuration ----------

SAMPLE_RATE = 16_000          # 16 kHz is sufficient for mosquito wingbeats
WINDOW_SECONDS = 1.0          # Process in 1-second chunks
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SECONDS = 0.5             # 50% overlap
HOP_SIZE = int(SAMPLE_RATE * HOP_SECONDS)

# Mosquito wingbeat fundamental frequency ranges (Hz):
# - Aedes aegypti female:    400-600 Hz
# - Aedes albopictus female: 450-650 Hz (overlapping)
# - Anopheles gambiae:       300-500 Hz
# - Culex pipiens:           300-400 Hz
# Source: Mukundarajan et al. eLife 2017, "Using mobile phones as acoustic sensors
# for high-throughput mosquito surveillance"
MOSQUITO_FREQ_MIN_HZ = 250
MOSQUITO_FREQ_MAX_HZ = 800

DETECTION_THRESHOLD = 0.65    # Confidence threshold for "mosquito present"
SPECIES_CONFIDENCE_MIN = 0.55 # Below this we say "unidentified mosquito"


# ---------- Feature extraction ----------


def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a feature vector from a 1-second audio window.

    Features (lightweight, no librosa dependency to keep Pi build small):
    1. Power spectrum bins in the mosquito frequency range (32 bins)
    2. Spectral centroid
    3. Spectral rolloff
    4. Zero-crossing rate
    5. RMS energy

    Total: 36 features. Small enough to run a lightweight model on Pi Zero.
    """
    if np is None:
        raise RuntimeError("numpy not available")

    # Bandpass filter to mosquito frequency range to suppress speech, motors, wind
    sos = scipy_signal.butter(
        4,
        [MOSQUITO_FREQ_MIN_HZ, MOSQUITO_FREQ_MAX_HZ],
        btype="bandpass",
        fs=sr,
        output="sos",
    )
    filtered = scipy_signal.sosfilt(sos, audio)

    # FFT
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / sr)

    # Power in 32 bins across 250-800 Hz
    band_mask = (freqs >= MOSQUITO_FREQ_MIN_HZ) & (freqs <= MOSQUITO_FREQ_MAX_HZ)
    band_fft = fft[band_mask]
    band_freqs = freqs[band_mask]
    if len(band_fft) < 32:
        band_features = np.pad(band_fft, (0, 32 - len(band_fft)))[:32]
    else:
        # Aggregate into 32 bins
        bin_edges = np.linspace(0, len(band_fft), 33).astype(int)
        band_features = np.array([
            band_fft[bin_edges[i]:bin_edges[i + 1]].mean()
            for i in range(32)
        ])

    # Spectral centroid (within mosquito band)
    if band_fft.sum() > 0:
        centroid = (band_freqs * band_fft).sum() / band_fft.sum()
    else:
        centroid = 0.0

    # Spectral rolloff (90%)
    cumsum = np.cumsum(band_fft)
    if cumsum[-1] > 0:
        rolloff_idx = np.searchsorted(cumsum, 0.9 * cumsum[-1])
        rolloff = band_freqs[min(rolloff_idx, len(band_freqs) - 1)]
    else:
        rolloff = 0.0

    # Zero-crossing rate
    zcr = ((audio[:-1] * audio[1:]) < 0).mean()

    # RMS energy
    rms = np.sqrt((audio ** 2).mean())

    return np.concatenate([
        band_features,
        [centroid, rolloff, zcr, rms]
    ]).astype(np.float32)


# ---------- Classifier ----------


class MosquitoClassifier:
    """
    Wraps a TFLite model. For Day 1 we ship a lightweight rule-based stand-in
    that detects mosquito presence by spectral peak in the wingbeat band.

    Person C replaces this with a real TFLite model trained on HumBug data.
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self.tflite_model = None
        if model_path and model_path.exists():
            self._load_tflite(model_path)
        else:
            log.warning(
                "No TFLite model at %s — using rule-based stand-in. "
                "Person C: train on HumBug dataset and save to acoustic/models/.",
                model_path,
            )

    def _load_tflite(self, path: Path):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                log.warning("TFLite runtime not available — using rule-based stand-in")
                return
        self.tflite_model = tflite.Interpreter(model_path=str(path))
        self.tflite_model.allocate_tensors()
        log.info("Loaded TFLite model from %s", path)

    def classify(self, audio: np.ndarray) -> dict:
        """Return {'mosquito_present': bool, 'confidence': float, 'species': str}"""
        features = extract_features(audio)

        if self.tflite_model is not None:
            return self._classify_tflite(features)
        else:
            return self._classify_rule_based(features, audio)

    def _classify_tflite(self, features: np.ndarray) -> dict:
        """Run inference with the loaded TFLite model."""
        input_details = self.tflite_model.get_input_details()
        output_details = self.tflite_model.get_output_details()
        self.tflite_model.set_tensor(input_details[0]["index"], features.reshape(1, -1))
        self.tflite_model.invoke()
        probs = self.tflite_model.get_tensor(output_details[0]["index"])[0]
        species_labels = ["aedes_aegypti", "aedes_albopictus", "anopheles", "culex", "noise"]
        top_idx = int(np.argmax(probs))
        return {
            "mosquito_present": species_labels[top_idx] != "noise"
                and float(probs[top_idx]) > DETECTION_THRESHOLD,
            "confidence": float(probs[top_idx]),
            "species": species_labels[top_idx]
                if float(probs[top_idx]) > SPECIES_CONFIDENCE_MIN
                else "unidentified_mosquito",
            "all_probs": {label: float(p) for label, p in zip(species_labels, probs)},
        }

    def _classify_rule_based(self, features: np.ndarray, audio: np.ndarray) -> dict:
        """
        Fallback: detect mosquito by sustained spectral peak in 400-600 Hz band.
        Used until Person C trains the real model.
        """
        band_features = features[:32]
        rms = features[-1]

        peak_power = band_features.max() if len(band_features) else 0.0
        mean_power = band_features.mean() if len(band_features) else 0.0
        peak_to_mean = peak_power / (mean_power + 1e-9)

        is_mosquito = peak_to_mean > 5.0 and rms > 0.005
        confidence = min(0.99, peak_to_mean / 20.0) if is_mosquito else 0.1

        return {
            "mosquito_present": is_mosquito,
            "confidence": float(confidence),
            "species": "unidentified_mosquito" if is_mosquito else "noise",
            "all_probs": {},
        }


# ---------- Real-time loop ----------


def run_realtime(model_path: Path | None = None, device: int | None = None):
    """Continuously classify audio from the default microphone."""
    if sd is None:
        raise RuntimeError(
            "sounddevice not installed — run on a Pi with audio capture support."
        )

    classifier = MosquitoClassifier(model_path)
    rolling_buffer = deque(maxlen=WINDOW_SIZE)
    last_result_time = 0.0

    log.info(
        "Listening on device=%s, sample_rate=%d Hz, window=%.1fs",
        device, SAMPLE_RATE, WINDOW_SECONDS,
    )

    def callback(indata, frames, time_info, status):
        nonlocal last_result_time
        if status:
            log.warning("Audio stream status: %s", status)
        rolling_buffer.extend(indata[:, 0])

        now = time.time()
        if len(rolling_buffer) >= WINDOW_SIZE and (now - last_result_time) >= HOP_SECONDS:
            audio = np.array(rolling_buffer, dtype=np.float32)
            result = classifier.classify(audio)
            last_result_time = now
            if result["mosquito_present"]:
                log.info(
                    "🦟 DETECTED: %s (%.0f%% confidence)",
                    result["species"], result["confidence"] * 100,
                )
            else:
                # Quiet log so the demo console doesn't spam
                pass

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=HOP_SIZE,
        device=device,
        callback=callback,
    ):
        log.info("Press Ctrl-C to stop")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            log.info("Stopped")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path(__file__).parent / "models/mosquito.tflite")
    p.add_argument("--device", type=int, default=None,
                   help="Audio input device index (use `python3 -c 'import sounddevice; print(sounddevice.query_devices())'`)")
    args = p.parse_args()
    run_realtime(args.model, args.device)
