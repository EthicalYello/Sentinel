# Sentinel Acoustic Trap

The hardware demo. One physical Raspberry Pi running real-time mosquito wingbeat classification on stage.

## What this is, honestly

A **research prototype**, not a production European Aedes albopictus classifier. Trained on the open HumBug Aedes aegypti dataset because that's what's available. The wingbeat signatures of Aedes aegypti and Aedes albopictus overlap substantially (both ~400–600 Hz), so the demo is honest if framed correctly:

> *"Here's the working sensing prototype, classifying mosquito wingbeats in real time. We trained it on Aedes aegypti — the species with the largest open-source labelled dataset. Producing a deployment-grade European classifier requires labelled field data, exactly the kind of work our LSHTM Logan Group partnership would unlock."*

This framing is a feature of the pitch, not a bug. It tells judges you understand the gap between research-grade and deployment-grade.

## Hardware bill of materials

| Item | Approx cost (UK) | Notes |
|------|------------------|-------|
| Raspberry Pi Zero 2W | £18 | Quad-core ARM, 512 MB RAM — sufficient |
| microSD card 32 GB | £8 | Class 10 |
| USB lavalier microphone (e.g. Boya BY-LM10) | £15 | Anything with a flat response below 1 kHz |
| OLED display (SSD1306, 128×64, I2C) | £8 | For visible "🦟 DETECTED" output |
| 5V 2.5A USB-C power supply | £10 | |
| Plastic enclosure | £8 | |
| Mini speaker for demo trigger (optional) | £10 | Plays an Aedes audio sample |
| Jumpers, USB-C OTG adapter | £5 | |
| **Total** | **~£82** | Under the typical T4PF expense allowance |

**Order today.** UK Amazon Prime delivery is your friend. Don't lose Day 1 to shipping.

## Setup steps (Person C, Day 1–2)

### 1. Flash the Pi

Use Raspberry Pi Imager to flash Raspberry Pi OS Lite (64-bit). Configure SSH, Wi-Fi, and your hostname before flashing — saves time later.

### 2. First boot

```bash
ssh pi@sentinel-trap.local
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-pip portaudio19-dev libatlas-base-dev git
```

### 3. Audio test

Plug in the USB microphone, then:

```bash
arecord -l                              # confirm USB mic is listed
arecord -D plughw:1,0 -d 5 test.wav     # record 5 seconds (adjust device index)
aplay test.wav                          # play it back if you have audio out
```

### 4. Install the classifier

```bash
git clone <repo-url> sentinel
cd sentinel/acoustic
pip3 install numpy scipy sounddevice tflite-runtime
```

### 5. Run it (rule-based fallback)

This works immediately — no model needed. Detects mosquito presence by spectral peak in the wingbeat band:

```bash
python3 classifier.py
```

You should see "🦟 DETECTED" lines when a mosquito sample plays nearby.

### 6. Train the real classifier (Day 5–7)

Download HumBug data and train a small TFLite model:

```bash
# On a laptop / desktop, not the Pi
mkdir -p training && cd training
# HumBug dataset: https://humbug.ox.ac.uk/data
# Or use the Abuzz dataset on Zenodo: https://zenodo.org/records/1217648

# Train script — Person C writes this; ~2 days work
python train_tflite.py --data humbug/ --out ../models/mosquito.tflite
```

Copy `mosquito.tflite` to the Pi:

```bash
scp models/mosquito.tflite pi@sentinel-trap.local:~/sentinel/acoustic/models/
```

Re-run the classifier — it will pick up the model automatically.

## Demo plan

For the 5-minute pitch:

1. **Pi sits on the demo table**, OLED screen visible to the audience.
2. **Speaker plays a recorded mosquito sample** (Aedes aegypti from HumBug) on cue.
3. **Within 1–2 seconds**, OLED screen displays: `🦟 DETECTED — Aedes — 87%`.
4. **Pitch presenter explains:** "This sensor costs under £15 to manufacture at scale. Imagine 10,000 of these scattered across European municipalities. That's the missing layer in the surveillance pipeline. Today's demo is one node. Production is everywhere."

## Backup plan

If the live classifier misbehaves on stage (background noise, audio driver glitch, etc.), have a **video of it working** queued in your slide deck. *Always* have a video. Demo gods are cruel.

## Common gotchas

- **USB microphone not detected:** check `lsusb` and `arecord -l`. Some lavalier mics need a powered USB hub.
- **Sample rate errors:** Pi Zero's USB audio sometimes stutters at 48 kHz. The code uses 16 kHz, which is more reliable.
- **tflite-runtime install fails:** on 64-bit Pi OS use the wheels from <https://google-coral.github.io/py-repo/>.
- **OLED screen blank:** SSD1306 uses I2C address 0x3C. Run `sudo i2cdetect -y 1` to confirm the Pi sees it.
- **Background noise triggers false positives:** lower the rule-based stand-in's threshold in `classifier.py` (`peak_to_mean > 5.0` → try `> 8.0`).
