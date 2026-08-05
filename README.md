Face Recognition

Face detection and identification for the office robot. Runs on a Jetson Orin Nano and lets the robot recognize returning visitors and greet them by name.

How it works

A two-stage pipeline:

Detection — YOLOv8n locates faces in each frame (TensorRT FP16 engine for on-device speed).
Recognition — SFace generates a 128-d embedding per face, which is matched by cosine similarity against enrolled identities. Anything below the similarity threshold is returned as unknown.

The loop runs at 10–20 Hz so downstream consumers (greeting logic, follow-me navigation) get low-latency identity updates.

Setup
bash
pip install -r requirements.txt
python build_engine.py        # exports YOLOv8n to a TensorRT FP16 engine

Model weights are expected in models/. The TensorRT engine is device-specific — rebuild it if you move to different hardware.

Usage

Enroll a new person:

bash
python enroll.py --name "Jane Doe" --images data/jane/

Run recognition on the live camera feed:

bash
python recognize.py --source 0
Configuration
Setting	Default	Notes
SIMILARITY_THRESHOLD	0.36	Raise to reduce false matches, lower to reduce misses
MIN_FACE_SIZE	40 px	Skips faces too small to embed reliably
TARGET_FPS	15	Perception loop rate
Notes
Embeddings are stored locally in data/embeddings.db; no face images or identity data leave the device.
Accuracy drops in strong backlighting — front-facing light on the visitor helps a lot.
3–5 enrollment images per person from slightly different angles works better than a single shot.
License

MIT
