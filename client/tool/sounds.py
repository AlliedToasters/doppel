"""Synthesize the Doppel skin's original sound effects (short, punchy, no samples). Run from repo root."""
import wave
from pathlib import Path
import numpy as np

SR = 44100
out = Path("client/assets/sounds"); out.mkdir(parents=True, exist_ok=True)
t = lambda d: np.arange(int(SR * d)) / SR

def env(x, a=0.003, d=0.08):
    tt = np.arange(len(x)) / SR
    return x * np.minimum(tt / a, 1) * np.exp(-tt / d)

def save(name, x):
    x = x / max(1e-9, np.abs(x).max()) * 0.85
    with wave.open(str(out / f"{name}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())

rng = np.random.default_rng(1)
tt = t(0.09)  # move: wooden "tok" with a pitch drop
f = 520 * np.exp(-tt * 18) + 260
save("move", env(np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.25 * rng.standard_normal(len(tt)) * np.exp(-tt * 120), d=0.03))
tt = t(0.22)  # capture: low thump + crunchy noise burst
f = 180 * np.exp(-tt * 14) + 70
x = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.6 * rng.standard_normal(len(tt)) * np.exp(-tt * 45)
save("capture", env(x, d=0.07))
tt = t(0.32)  # check: two quick rising blips
x = np.zeros(len(tt))
for k, (st, fr) in enumerate(((0.0, 880), (0.09, 1320))):
    s = int(st * SR); seg = t(0.12)
    x[s:s + len(seg)] += env(np.sign(np.sin(2 * np.pi * fr * seg)) * 0.4 + np.sin(2 * np.pi * fr * seg), d=0.05)
save("check", x)
tt = t(1.1)  # thunder: crack + rumble, for the queen-capture lightning
crack = rng.standard_normal(len(tt)) * np.exp(-tt * 9)
rumble = np.convolve(rng.standard_normal(len(tt)), np.ones(400) / 400, "same") * 6 * np.exp(-tt * 2.5)
save("thunder", env(crack + rumble, a=0.001, d=0.5))
tt = t(0.9)  # win: rising arpeggio
x = np.zeros(len(tt))
for k, fr in enumerate((523, 659, 784, 1047)):
    s = int(k * 0.11 * SR); seg = t(0.35)
    x[s:s + len(seg)] += env(np.sin(2 * np.pi * fr * seg) + 0.3 * np.sin(4 * np.pi * fr * seg), d=0.18)[: len(x) - s]
save("win", x)
print("ok")
