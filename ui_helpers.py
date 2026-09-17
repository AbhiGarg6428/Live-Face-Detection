"""
UI & HUD Helpers for OpenCV Video Stream
Provides sleek HUD overlays, corner-bracket bounding boxes, and dialogs.
"""

import cv2
import numpy as np

# Color Palette (BGR)
COLOR_BG_DARK = (20, 20, 24)
COLOR_ACCENT_GREEN = (72, 210, 108)    # Match / Recognition (Emerald)
COLOR_ACCENT_CYAN = (235, 185, 45)     # Registration / Capture (Cyan)
COLOR_ACCENT_RED = (80, 80, 245)       # Unknown / Alert (Crimson)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 140, 140)
COLOR_DARK_OVERLAY = (15, 15, 18)


def draw_corner_rect(frame, bbox, color, thickness=2, corner_len=18):
    """
    Draws a sleek bounding box with highlighted tech corner brackets.
    """
    x, y, w, h = bbox
    # Draw subtle background box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

    # Top-Left
    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness, cv2.LINE_AA)

    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness, cv2.LINE_AA)

    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness, cv2.LINE_AA)

    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness, cv2.LINE_AA)


def draw_badge(frame, text, pos, bg_color, text_color=COLOR_WHITE, font_scale=0.55):
    """
    Draws a clean filled badge with text at (x, y).
    """
    x, y = pos
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)

    pad_x, pad_y = 10, 6
    bx1 = x
    by1 = y - th - pad_y * 2
    bx2 = x + tw + pad_x * 2
    by2 = y

    # Clamp within screen
    h_frame, w_frame = frame.shape[:2]
    if by1 < 0:
        by1 = y + 5
        by2 = y + th + pad_y * 2 + 5
        text_y = by1 + th + pad_y
    else:
        text_y = y - pad_y

    sub_img = frame[max(0, by1):min(h_frame, by2), max(0, bx1):min(w_frame, bx2)]
    if sub_img.size > 0:
        rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.25, rect, 0.75, 1.0, sub_img)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (bx1 + pad_x, text_y), font, font_scale, text_color, 1, cv2.LINE_AA)


def draw_hud(frame, mode: str, fps: float, enrolled_count: int, threshold: float):
    """
    Renders top status bar and bottom keybindings guide.
    """
    h, w = frame.shape[:2]

    # Top Header Bar (semi-transparent)
    top_bar_h = 50
    header_roi = frame[0:top_bar_h, 0:w]
    dark_mask = np.full(header_roi.shape, (12, 12, 16), dtype=np.uint8)
    cv2.addWeighted(header_roi, 0.35, dark_mask, 0.65, 0, header_roi)
    cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (50, 50, 60), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_DUPLEX

    # Left: Mode Pill
    if mode == "RECOGNIZE":
        mode_text = "[MODE 1: LIVE RECOGNITION]"
        pill_color = COLOR_ACCENT_GREEN
    else:
        mode_text = "[MODE 2: FACE ENROLLMENT]"
        pill_color = COLOR_ACCENT_CYAN

    cv2.putText(frame, mode_text, (20, 32), font, 0.65, pill_color, 2, cv2.LINE_AA)

    # Right: System telemetry
    stats_text = f"FPS: {fps:4.1f}   |   Enrolled: {enrolled_count}   |   Thresh: {threshold:.2f}"
    (tw, _), _ = cv2.getTextSize(stats_text, font, 0.5, 1)
    cv2.putText(frame, stats_text, (w - tw - 20, 32), font, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # Bottom Footer Bar (Keyboard shortcuts)
    footer_h = 42
    footer_roi = frame[h - footer_h:h, 0:w]
    footer_mask = np.full(footer_roi.shape, (12, 12, 16), dtype=np.uint8)
    cv2.addWeighted(footer_roi, 0.35, footer_mask, 0.65, 0, footer_roi)
    cv2.line(frame, (0, h - footer_h), (w, h - footer_h), (50, 50, 60), 1, cv2.LINE_AA)

    guide_text = "[1] Recognize  |  [2] Store  |  [SPACE] Capture  |  [H] Landmarks  |  [L] List  |  [D] Delete  |  [Q] Quit"
    (gw, _), _ = cv2.getTextSize(guide_text, font, 0.45, 1)
    cv2.putText(frame, guide_text, (int((w - gw) / 2), h - 14), font, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)


def draw_notification(frame, message: str, level="INFO"):
    """
    Draws a centered banner notification across the screen for important events.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    if level == "SUCCESS":
        color = COLOR_ACCENT_GREEN
    elif level == "WARNING":
        color = COLOR_ACCENT_CYAN
    elif level == "DANGER":
        color = COLOR_ACCENT_RED
    else:
        color = (180, 180, 180)

    (tw, th), _ = cv2.getTextSize(message, font, 0.65, 1)
    pad = 12
    box_y1 = int(h / 2) - th - pad
    box_y2 = int(h / 2) + pad
    box_x1 = int((w - tw) / 2) - pad * 2
    box_x2 = int((w + tw) / 2) + pad * 2

    # Draw darkened backdrop
    sub = frame[box_y1:box_y2, box_x1:box_x2]
    if sub.size > 0:
        dark = np.full(sub.shape, (10, 10, 12), dtype=np.uint8)
        cv2.addWeighted(sub, 0.2, dark, 0.8, 0, sub)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 2, cv2.LINE_AA)
        cv2.putText(frame, message, (int((w - tw) / 2), int(h / 2)), font, 0.65, color, 1, cv2.LINE_AA)


def prompt_name_dialog(title="Register New Face", prompt="Enter the full name for this person:"):
    """
    Opens a clean native input dialog using Tkinter so the user can easily type.
    """
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    name = simpledialog.askstring(title, prompt, parent=root)
    root.destroy()
    return name.strip() if name else None


def prompt_select_dialog(title="Select Face to Delete", options=None):
    """
    Simple dialog to choose a face name from registered people.
    """
    if not options:
        return None
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    prompt_str = "Type name to delete:\n" + ", ".join(options)
    name = simpledialog.askstring(title, prompt_str, parent=root)
    root.destroy()
    return name.strip() if name else None
