"""
Live Face Recognition & Enrollment System
Modes:
  1. RECOGNIZE MODE: Real-time face detection & identity matching with confidence scores.
  2. STORE MODE: Captures face samples, prompts for person's name, and registers to database.

Controls:
  [1]     : Switch to Recognition Mode
  [2]     : Switch to Face Enrollment Mode
  [SPACE] : Capture & Register Face (in Mode 2)
  [L]     : List Registered Faces
  [D]     : Delete a Registered Face
  [+] / [-]: Adjust Recognition Confidence Threshold
  [Q]/ESC : Exit
"""

import sys
import time
import os
import cv2
import numpy as np

from face_engine import FaceEngine
import ui_helpers


def play_sound(sound_type="beep"):
    """Plays subtle audio feedback on Windows without blocking."""
    try:
        import winsound
        if sound_type == "beep":
            winsound.Beep(1200, 70)
        elif sound_type == "success":
            winsound.Beep(800, 90)
            winsound.Beep(1200, 140)
        elif sound_type == "warn":
            winsound.Beep(400, 150)
    except Exception:
        pass


def open_camera(preferred_idx=0):
    """Opens webcam with DirectShow backend for faster startup on Windows."""
    cap = cv2.VideoCapture(preferred_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(preferred_idx)
    if not cap.isOpened():
        return None

    # Request high definition, camera will adapt to highest supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def main():
    print("=" * 60)
    print("  LIVE FACE RECOGNITION & REGISTRATION SYSTEM")
    print("=" * 60)
    print("Initializing Face Engine (YuNet + SFace)...")

    try:
        engine = FaceEngine(
            conf_threshold=0.6,
            nms_threshold=0.3,
            cosine_threshold=0.38,
        )
    except Exception as e:
        print(f"[FATAL] Failed to initialize engine: {e}")
        return

    enrolled = engine.list_people()
    print(f"Loaded {len(enrolled)} enrolled identities from database.")
    for p in enrolled:
        print(f"  - {p['name']} ({p['samples']} samples, added {p['date']})")

    print("\nOpening camera...")
    cap = open_camera(0)
    if cap is None:
        print("[ERROR] Could not open webcam. Please check if another app is using it.")
        return

    window_name = "Face Recognition System - [1] Recognize | [2] Store"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    mode = "RECOGNIZE"  # "RECOGNIZE" or "STORE"
    notification_msg = "Press [1] to Recognize | [2] to Store Face"
    notification_level = "INFO"
    notification_expire_time = time.time() + 4.0

    # Multi-sample capture state for enrollment
    capture_in_progress = False
    target_person_name = ""
    collected_features = []
    collected_crops = []
    target_samples = 6
    sample_interval = 0.25  # seconds between samples
    last_sample_time = 0.0

    show_landmarks = False  # Blue landmark dots hidden by default

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to read camera frame.")
            time.sleep(0.05)
            continue

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Calculate FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Detect faces
        detected_faces = engine.detect_faces(frame)

        # --- MODE 1: LIVE RECOGNITION ---
        if mode == "RECOGNIZE":
            for face_info in detected_faces:
                raw_face = face_info["raw"]
                bbox = face_info["bbox"]
                landmarks = face_info["landmarks"]

                # Extract feature & match
                feature, _ = engine.extract_feature(frame, raw_face)
                name, score, is_match = engine.match_face(feature)

                sim_pct = int(min(100, max(0, (score / 0.7) * 100)))  # Scaled cosine similarity %

                if is_match:
                    color = ui_helpers.COLOR_ACCENT_GREEN
                    label = f"{name} ({sim_pct}%)"
                else:
                    color = ui_helpers.COLOR_ACCENT_RED
                    label = f"Unknown ({sim_pct}%)"

                # Draw tech-style bounding box
                ui_helpers.draw_corner_rect(frame, bbox, color, thickness=2)

                # Draw landmark points (hidden by default)
                if show_landmarks:
                    for _, pt in landmarks.items():
                        cv2.circle(frame, pt, 2, ui_helpers.COLOR_ACCENT_CYAN, -1, cv2.LINE_AA)

                # Draw name badge above box
                badge_x = bbox[0]
                badge_y = bbox[1] - 4
                ui_helpers.draw_badge(frame, label, (badge_x, badge_y), color)

        # --- MODE 2: FACE ENROLLMENT / STORE ---
        elif mode == "STORE":
            # Draw enrollment guide box in center
            guide_w, guide_h = int(w * 0.4), int(h * 0.55)
            gx1 = int((w - guide_w) / 2)
            gy1 = int((h - guide_h) / 2)
            gx2, gy2 = gx1 + guide_w, gy1 + guide_h

            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (60, 60, 70), 1, cv2.LINE_AA)
            ui_helpers.draw_corner_rect(frame, [gx1, gy1, guide_w, guide_h], ui_helpers.COLOR_ACCENT_CYAN, thickness=2)

            if not capture_in_progress:
                # Instruction banner
                instruct_txt = "Position face in center & press [SPACE] to Register"
                font = cv2.FONT_HERSHEY_DUPLEX
                (iw, _), _ = cv2.getTextSize(instruct_txt, font, 0.55, 1)
                cv2.putText(frame, instruct_txt, (int((w - iw) / 2), gy2 + 35), font, 0.55, ui_helpers.COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

                # Highlight detected face inside guide
                for face_info in detected_faces:
                    bbox = face_info["bbox"]
                    ui_helpers.draw_corner_rect(frame, bbox, ui_helpers.COLOR_WHITE, thickness=1)
            else:
                # Actively capturing samples
                if len(detected_faces) == 1:
                    face_info = detected_faces[0]
                    raw_face = face_info["raw"]
                    bbox = face_info["bbox"]
                    ui_helpers.draw_corner_rect(frame, bbox, ui_helpers.COLOR_ACCENT_CYAN, thickness=3)

                    if curr_time - last_sample_time >= sample_interval:
                        last_sample_time = curr_time
                        feat, crop = engine.extract_feature(frame, raw_face)
                        collected_features.append(feat)
                        collected_crops.append(crop)
                        play_sound("beep")

                        if len(collected_features) >= target_samples:
                            # Registration complete! Save to database
                            engine.register_person(target_person_name, collected_features, collected_crops)
                            play_sound("success")
                            notification_msg = f"Successfully registered: {target_person_name}!"
                            notification_level = "SUCCESS"
                            notification_expire_time = time.time() + 4.5
                            capture_in_progress = False
                            mode = "RECOGNIZE"  # Return to recognition mode
                else:
                    # Face lost during capture sequence
                    pass

                # Draw progress bar
                progress = len(collected_features) / float(target_samples)
                bar_w = int(w * 0.5)
                bx = int((w - bar_w) / 2)
                by = gy2 + 25
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 16), (40, 40, 50), -1)
                fill_w = int(bar_w * progress)
                cv2.rectangle(frame, (bx, by), (bx + fill_w, by + 16), ui_helpers.COLOR_ACCENT_CYAN, -1)
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 16), ui_helpers.COLOR_WHITE, 1)

                prog_text = f"Enrolling {target_person_name}: {len(collected_features)}/{target_samples} samples (move slightly)"
                font = cv2.FONT_HERSHEY_DUPLEX
                (pw, _), _ = cv2.getTextSize(prog_text, font, 0.5, 1)
                cv2.putText(frame, prog_text, (int((w - pw) / 2), by - 8), font, 0.5, ui_helpers.COLOR_WHITE, 1, cv2.LINE_AA)

        # Draw HUD elements
        enrolled_count = len(engine.database)
        ui_helpers.draw_hud(frame, mode, fps, enrolled_count, engine.cosine_threshold)

        # Draw active notifications
        if time.time() < notification_expire_time:
            ui_helpers.draw_notification(frame, notification_msg, notification_level)

        cv2.imshow(window_name, frame)

        # Handle Keyboard Input
        key = cv2.waitKey(1) & 0xFF

        # Exit
        if key in (ord("q"), ord("Q"), 27):  # 27 is ESC
            print("\nExiting Face Recognition System...")
            break

        # Mode 1: Switch to Recognition Mode
        elif key == ord("1"):
            mode = "RECOGNIZE"
            capture_in_progress = False
            notification_msg = "Switched to Mode 1: Live Recognition"
            notification_level = "INFO"
            notification_expire_time = time.time() + 2.5
            print("[INFO] Switched to Recognition Mode.")

        # Mode 2: Switch to Enrollment / Store Mode
        elif key == ord("2"):
            mode = "STORE"
            capture_in_progress = False
            notification_msg = "Mode 2: Press [SPACE] to capture & store face"
            notification_level = "WARNING"
            notification_expire_time = time.time() + 3.0
            print("[INFO] Switched to Enrollment / Store Mode.")

        # Trigger Face Registration in Mode 2
        elif key == 32:  # SPACE bar
            if mode != "STORE":
                mode = "STORE"
                notification_msg = "Switched to Store Mode. Press [SPACE] to capture."
                notification_level = "WARNING"
                notification_expire_time = time.time() + 2.5
            else:
                if len(detected_faces) == 0:
                    play_sound("warn")
                    notification_msg = "No face detected! Please face the camera."
                    notification_level = "DANGER"
                    notification_expire_time = time.time() + 3.0
                elif len(detected_faces) > 1:
                    play_sound("warn")
                    notification_msg = "Multiple faces detected! Please enroll one at a time."
                    notification_level = "DANGER"
                    notification_expire_time = time.time() + 3.0
                else:
                    # Prompt for person's name
                    entered_name = ui_helpers.prompt_name_dialog()
                    if entered_name:
                        target_person_name = entered_name
                        collected_features = []
                        collected_crops = []
                        capture_in_progress = True
                        last_sample_time = curr_time
                        notification_msg = f"Look at camera and tilt head slightly..."
                        notification_level = "WARNING"
                        notification_expire_time = time.time() + 3.0
                    else:
                        notification_msg = "Registration cancelled."
                        notification_level = "INFO"
                        notification_expire_time = time.time() + 2.0

        # List enrolled people
        elif key in (ord("l"), ord("L")):
            people = engine.list_people()
            print("\n" + "=" * 40)
            print(f"  REGISTERED IDENTITIES ({len(people)} total):")
            print("=" * 40)
            if not people:
                print("  No faces registered yet. Press '2' to store.")
            for p in people:
                print(f"  * {p['name']} ({p['samples']} samples, enrolled {p['date']})")
            print("=" * 40)
            names_str = ", ".join([p["name"] for p in people]) if people else "None"
            notification_msg = f"Enrolled: {names_str}"
            notification_level = "INFO"
            notification_expire_time = time.time() + 4.0

        # Delete an enrolled person
        elif key in (ord("d"), ord("D")):
            people = engine.list_people()
            if not people:
                notification_msg = "No enrolled faces to delete."
                notification_level = "INFO"
                notification_expire_time = time.time() + 2.5
            else:
                names = [p["name"] for p in people]
                del_name = ui_helpers.prompt_select_dialog("Delete Face Record", names)
                if del_name:
                    if engine.delete_person(del_name):
                        play_sound("warn")
                        notification_msg = f"Deleted record for: {del_name}"
                        notification_level = "WARNING"
                        notification_expire_time = time.time() + 3.0
                        print(f"[INFO] Deleted face record: {del_name}")
                    else:
                        notification_msg = f"Person '{del_name}' not found."
                        notification_level = "DANGER"
                        notification_expire_time = time.time() + 3.0

        # Adjust threshold (+ / -)
        elif key in (ord("+"), ord("=")):
            engine.cosine_threshold = min(0.60, engine.cosine_threshold + 0.02)
            notification_msg = f"Cosine Threshold: {engine.cosine_threshold:.2f}"
            notification_level = "INFO"
            notification_expire_time = time.time() + 2.0
        elif key in (ord("-"), ord("_")):
            engine.cosine_threshold = max(0.20, engine.cosine_threshold - 0.02)
            notification_msg = f"Cosine Threshold: {engine.cosine_threshold:.2f}"
            notification_level = "INFO"
            notification_expire_time = time.time() + 2.0

        # Toggle landmarks visibility
        elif key in (ord("h"), ord("H")):
            show_landmarks = not show_landmarks
            notification_msg = f"Landmarks: {'ON' if show_landmarks else 'OFF'}"
            notification_level = "INFO"
            notification_expire_time = time.time() + 2.0

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()
