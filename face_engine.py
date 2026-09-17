"""
Face Recognition & Detection Engine
Uses OpenCV's YuNet for ultra-fast face detection and SFace for 128-D feature embeddings.
"""

import os
import json
import pickle
import numpy as np
import cv2

DEFAULT_YUNET_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
DEFAULT_SFACE_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")
DEFAULT_DB_PATH = os.path.join("data", "faces_db.pkl")
DEFAULT_METADATA_PATH = os.path.join("data", "faces_db.json")
DEFAULT_FACES_DIR = os.path.join("data", "registered_faces")


class FaceEngine:
    def __init__(
        self,
        yunet_path=DEFAULT_YUNET_PATH,
        sface_path=DEFAULT_SFACE_PATH,
        db_path=DEFAULT_DB_PATH,
        faces_dir=DEFAULT_FACES_DIR,
        conf_threshold=0.6,
        nms_threshold=0.3,
        cosine_threshold=0.38,
    ):
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.db_path = db_path
        self.metadata_path = DEFAULT_METADATA_PATH
        self.faces_dir = faces_dir
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.cosine_threshold = cosine_threshold

        self.current_input_size = (320, 320)
        self.detector = None
        self.recognizer = None

        # Database format:
        # {
        #     "Person Name": {
        #         "features": [np.ndarray(shape=(1, 128)), ...],
        #         "registered_at": "timestamp",
        #         "sample_count": 5
        #     }
        # }
        self.database = {}

        self._init_models()
        self._load_database()

    def _init_models(self):
        """Initializes YuNet detector and SFace recognizer."""
        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"YuNet model not found at {self.yunet_path}")
        if not os.path.exists(self.sface_path):
            raise FileNotFoundError(f"SFace model not found at {self.sface_path}")

        # Initialize YuNet Face Detector
        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_path,
            "",
            self.current_input_size,
            self.conf_threshold,
            self.nms_threshold,
            5000,
        )

        # Initialize SFace Face Recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")

    def update_input_size(self, width: int, height: int):
        """Updates the input image resolution for the detector."""
        if self.current_input_size != (width, height):
            self.current_input_size = (width, height)
            self.detector.setInputSize((width, height))

    def detect_faces(self, frame):
        """
        Detects faces in the given BGR frame.
        Returns a list of face dicts containing bounding box, confidence, and landmarks.
        """
        h, w = frame.shape[:2]
        self.update_input_size(w, h)

        ret, raw_faces = self.detector.detect(frame)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        results = []
        for face in raw_faces:
            # face: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            bbox = [int(v) for v in face[0:4]]
            # Ensure coordinates are within frame
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(w - bbox[0], bbox[2])
            bbox[3] = min(h - bbox[1], bbox[3])

            landmarks = {
                "right_eye": (int(face[4]), int(face[5])),
                "left_eye": (int(face[6]), int(face[7])),
                "nose_tip": (int(face[8]), int(face[9])),
                "right_mouth": (int(face[10]), int(face[11])),
                "left_mouth": (int(face[12]), int(face[13])),
            }
            score = float(face[14])

            results.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "score": score,
                "raw": face,
            })
        return results

    def extract_feature(self, frame, raw_face):
        """
        Aligns the face based on 5 landmarks and extracts a 128-D L2-normalized feature embedding.
        """
        aligned_face = self.recognizer.alignCrop(frame, raw_face)
        feature = self.recognizer.feature(aligned_face)
        return feature, aligned_face

    def match_face(self, query_feature):
        """
        Matches a query feature vector against all enrolled identities in the database.
        Returns: (name, similarity_score, is_match)
        """
        if not self.database:
            return ("Unknown", 0.0, False)

        best_name = "Unknown"
        best_score = -1.0

        for name, profile in self.database.items():
            stored_features = profile.get("features", [])
            for feat in stored_features:
                # Cosine similarity matching
                score = self.recognizer.match(query_feature, feat, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_name = name

        is_match = best_score >= self.cosine_threshold
        final_name = best_name if is_match else "Unknown"
        return (final_name, max(0.0, float(best_score)), is_match)

    def register_person(self, name: str, feature_list: list, sample_crops: list = None):
        """
        Registers or updates a person's profile with one or multiple feature embeddings and preview crops.
        """
        name = name.strip()
        if not name:
            raise ValueError("Name cannot be empty.")

        import time
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)

        person_folder = os.path.join(self.faces_dir, name)
        os.makedirs(person_folder, exist_ok=True)

        # Save preview crops
        if sample_crops:
            for i, crop in enumerate(sample_crops):
                crop_path = os.path.join(person_folder, f"sample_{i+1}.jpg")
                cv2.imwrite(crop_path, crop)

        if name in self.database:
            # Append new features up to max 15 to avoid database bloat
            existing = self.database[name]["features"]
            combined = existing + feature_list
            self.database[name]["features"] = combined[-15:]
            self.database[name]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.database[name]["sample_count"] = len(self.database[name]["features"])
        else:
            self.database[name] = {
                "features": feature_list[:15],
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_count": len(feature_list),
            }

        self._save_database()
        return True

    def delete_person(self, name: str):
        """Deletes a person from the database and removes their stored crops."""
        if name in self.database:
            del self.database[name]
            self._save_database()

            # Clean up photos folder
            import shutil
            person_folder = os.path.join(self.faces_dir, name)
            if os.path.exists(person_folder):
                shutil.rmtree(person_folder, ignore_errors=True)
            return True
        return False

    def list_people(self):
        """Returns a list of all registered names and their sample counts."""
        return [
            {
                "name": name,
                "samples": data.get("sample_count", len(data.get("features", []))),
                "date": data.get("registered_at", "N/A"),
            }
            for name, data in self.database.items()
        ]

    def _save_database(self):
        """Saves embeddings to binary pickle and metadata to JSON."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)

        # Write human-readable metadata file
        metadata = {
            name: {
                "sample_count": data.get("sample_count", len(data.get("features", []))),
                "registered_at": data.get("registered_at", "N/A"),
            }
            for name, data in self.database.items()
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def _load_database(self):
        """Loads embeddings from pickle file if it exists."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.database = pickle.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load database from {self.db_path}: {e}")
                self.database = {}
        else:
            self.database = {}
