# Live Face Recognition & Registration System

A fast, lightweight, and accurate real-time facial recognition and enrollment application using Python, OpenCV, and deep-learning ONNX models (**YuNet** for detection + **SFace** for 128-D facial embeddings).

---

## Features

- **Mode 1: Live Recognition (`[1]`)**:
  - Real-time webcam face detection with landmark tracking (eyes, nose, mouth).
  - High-precision matching via Cosine Similarity against enrolled identities.
  - Custom HUD bounding boxes: **Green** with name and confidence percentage for recognized faces, **Red** with similarity score for unknown faces.
- **Mode 2: Face Enrollment & Data Storage (`[2]`)**:
  - Centered alignment target box for optimal capture.
  - Press `[SPACE]` to trigger registration.
  - Native popup dialog to enter the person's name.
  - Automatically captures 6 multi-angle face samples with visual progress bar and audio beeps.
  - Stores 128-D feature vectors in `data/faces_db.pkl` and saves cropped face previews in `data/registered_faces/<Name>/`.
- **Database Management**:
  - `[L]`: List all registered identities with sample counts and registration timestamps.
  - `[D]`: Delete any registered person.
  - `[+]` / `[-]`: Adjust Cosine Similarity threshold on the fly.
- **Audio Feedback**:
  - Subtle sound cues on sample capture, successful enrollment, and warnings.

---

## Directory Layout

```
d:/face recoginazion/
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx     # YuNet Face Detector (~232 KB)
│   └── face_recognition_sface_2021dec.onnx    # SFace Face Recognizer (~37 MB)
│
├── data/
│   ├── faces_db.pkl                          # Binary embeddings database
│   ├── faces_db.json                         # Human-readable metadata
│   └── registered_faces/                     # Cropped sample images per person
│       └── <Person_Name>/
│           ├── sample_1.jpg
│           └── ...
│
├── face_engine.py                            # Model loading, detection & recognition logic
├── ui_helpers.py                             # HUD graphics, badges, dialogs
├── main.py                                   # Main camera loop & mode manager
├── test_engine.py                            # Automated unit verification test
├── run.bat                                   # 1-Click launcher for Windows
└── requirements.txt                          # Dependencies
```

---

## How to Run

### Method 1: Double-Click (Recommended)
Double-click `run.bat` in File Explorer.

### Method 2: Command Line
Open PowerShell or Command Prompt in the project folder and run:
```powershell
py -3.13 main.py
```

---

## Keyboard Controls

| Key | Action |
| --- | --- |
| `1` | Switch to **Mode 1: Live Recognition** |
| `2` | Switch to **Mode 2: Face Enrollment / Store** |
| `SPACE` | Capture face samples and register (in Mode 2) |
| `H` | Toggle facial landmark dots (hidden by default) |
| `L` | List all enrolled faces in terminal and on-screen HUD |
| `D` | Delete an enrolled face record |
| `+` / `-` | Increase / decrease recognition confidence threshold |
| `Q` / `ESC` | Exit application |

---

## Tips for Best Recognition Accuracy

1. **Good Lighting**: Ensure your face is evenly illuminated (avoid strong backlighting).
2. **Natural Angle Movement during Registration**: When enrolling, tilt your head slightly (left, right, up, down) between the 6 sample beeps to capture slight angle variations.
3. **Distance**: Stay roughly 0.5 to 1.5 meters from the webcam.
