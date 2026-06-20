# ARES
**Autonomous Recognition and Engagement System**

Embedded targeting pipeline for free-fall munitions. ARES uses a custom, lightweight YOLOv8 architecture coupled with dynamic density clustering to lock onto high-value target concentrations in real-time.

Designed for edge inference.

## System Overview

*   **Detection Pipeline:** Custom YOLOv8 backbone (`ares_arge.yaml`) optimized for low-latency inference on edge hardware. Detects 3 core classes: `tank`, `soldier`, `vehicle`.
*   **Target Selection:** Extracts coordinates of all detected targets and applies dynamic K-Means clustering. The system automatically shifts focus to the cluster with the highest density.
*   **Tracking:** Uses an exponential moving average (EMA) filter to smoothly transition the targeting crosshair to the optimal centroid.
*   **Target Hardware:** Raspberry Pi 5, Arduino Mega, ESP32.

## Structure

*   `/ai`: Dataset handling, training pipelines, and raw inference scripts.
*   `/guidance`: Contains `guidance.py`, the core logic for spatial clustering and crosshair tracking.
*   `/models/arge`: Custom architecture configurations (e.g., modified YOLOv8 definitions).

## Running the Tracker

Install dependencies:
```bash
pip install ultralytics opencv-python scikit-learn numpy
```

Run the targeting loop:
```bash
python guidance/guidance.py
```
*Note: Hardcoded paths for weights and video sources in `guidance.py` must be adjusted to your local environment before execution.*

## Status
Active development. Model optimization and hardware IO integration in progress.
