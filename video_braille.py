#!/usr/bin/env python3
"""
Video -> (Color) Braille player in the terminal.

Features
- Decodes frames with OpenCV (preferred for simplicity and portability).
- Renders grayscale Braille or 24-bit color Braille using ANSI escape codes.
- Auto-resizes to a target character width; keeps aspect ratio suitable for terminal cells.
- Basic frame pacing using source FPS or a user limit.

Notes
- For Windows terminals, UTF-8 and ANSI are enabled via colorama if present.
- Audio is optional and external (e.g., run `ffplay -nodisp` in parallel).

Dependencies
- Required: numpy
- Recommended: opencv-python, colorama
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple, Any
import json
from datetime import datetime
import re
import shutil
import subprocess
import signal

import numpy as np

# Optional dependencies
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import colorama  # type: ignore
except Exception:  # pragma: no cover
    colorama = None


def enable_windows_utf8():
    if os.name == "nt":
        try:
            os.system("chcp 65001 > nul")
        except Exception:
            pass


def init_ansi():
    if colorama is not None:
        try:
            colorama.just_fix_windows_console()
        except Exception:
            # Fallback init if needed
            try:
                colorama.init()
            except Exception:
                pass


def hide_cursor():
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()


def show_cursor():
    sys.stdout.write("\x1b[?25h\x1b[0m")  # also reset any color
    sys.stdout.flush()


def home_and_clear_lines(height_chars: Optional[int] = None):
    # Move cursor to home. Avoid full clear for performance.
    sys.stdout.write("\x1b[H")
    if height_chars is not None:
        # Ensure previous frame's extra lines are cleared when size changes.
        for _ in range(height_chars):
            sys.stdout.write("\x1b[2K\n")
        sys.stdout.write("\x1b[H")
    sys.stdout.flush()


def compute_target_cell_size(frame_w: int, frame_h: int, width_chars: int, aspect: float = 0.5) -> tuple[int, int]:
    """Compute target Braille cell grid size (W,H) from source frame size and target width.

    Each Braille char covers a 2x4 pixel block after resize. Term characters are taller,
    so we apply a 0.5 vertical compensation factor to preserve aspect.
    """
    if frame_w <= 0 or frame_h <= 0:
        return width_chars, max(1, int(width_chars * 0.5))
    aspect_src = frame_h / max(1, frame_w)
    height_chars = max(1, int(width_chars * aspect_src * aspect))
    return width_chars, height_chars


def _build_mask(
    gray: np.ndarray,
    threshold: int,
    invert: bool,
    adaptive: str,
    block_size: int,
    adaptive_c: int,
    gamma: float,
    clahe_clip: Optional[float],
    clahe_grid: int,
    blur_kernel: int,
    edge_boost: bool,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    g = gray
    if invert:
        g = 255 - g
    # gamma correction (power law). gamma < 1 brightens mid-tones; > 1 darkens.
    if abs(gamma - 1.0) > 1e-6:
        f = g.astype(np.float32) / 255.0
        f = np.clip(np.power(f, gamma), 0.0, 1.0)
        g = (f * 255.0 + 0.5).astype(np.uint8)
    # CLAHE for local contrast
    if clahe_clip is not None and clahe_clip > 0:
        try:
            clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
            g = clahe.apply(g)
        except Exception:
            pass
    # Optional blur to reduce noise prior to thresholding
    if blur_kernel and blur_kernel >= 3 and blur_kernel % 2 == 1:
        g = cv2.GaussianBlur(g, (blur_kernel, blur_kernel), 0)

    # Thresholding
    if adaptive == "mean":
        bin_img = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, max(3, block_size | 1), adaptive_c)
        mask = bin_img > 0
    elif adaptive == "gaussian":
        bin_img = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, max(3, block_size | 1), adaptive_c)
        mask = bin_img > 0
    elif adaptive == "otsu":
        _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = bin_img > 0
    else:
        mask = g < threshold

    # Edge boost: OR in edges from Canny
    if edge_boost:
        try:
            edges = cv2.Canny(g, canny_low, canny_high)
            if edges is not None:
                # Dilate slightly to make edges visible in cell space
                edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                mask = mask | (edges > 0)
        except Exception:
            pass

    return mask


def frame_to_braille_gray(
    frame_bgr: np.ndarray,
    width_chars: int,
    threshold: int = 130,
    invert: bool = False,
    aspect: float = 0.5,
    adaptive: str = "none",
    block_size: int = 15,
    adaptive_c: int = 5,
    gamma: float = 1.0,
    clahe_clip: Optional[float] = None,
    clahe_grid: int = 8,
    blur_kernel: int = 0,
    edge_boost: bool = False,
    canny_low: int = 100,
    canny_high: int = 200,
) -> list[str]:
    """Convert a BGR frame to grayscale Braille text lines.

    Returns a list of strings (one per text row).
    """
    h, w = frame_bgr.shape[:2]
    Wc, Hc = compute_target_cell_size(w, h, width_chars, aspect=aspect)
    # Resize to pixel grid matching Braille cells (2x4 per char)
    target_w = max(2, Wc * 2)
    target_h = max(4, Hc * 4)
    small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Ensure dimensions are multiples of (4,2)
    H = (gray.shape[0] // 4) * 4
    W = (gray.shape[1] // 2) * 2
    if H == 0 or W == 0:
        return []
    g = gray[:H, :W]
    # Build boolean mask for dots according to selected preprocessing/thresholding
    mask_bool = _build_mask(
        g,
        threshold=threshold,
        invert=invert,
        adaptive=adaptive,
        block_size=block_size,
        adaptive_c=adaptive_c,
        gamma=gamma,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        blur_kernel=blur_kernel,
        edge_boost=edge_boost,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    # Build bitmask per cell using slicing (positions per 2x4 block)
    cells_h = H // 4
    cells_w = W // 2
    mask = np.zeros((cells_h, cells_w), dtype=np.uint16)

    def bit(cond: np.ndarray, value: int):
        nonlocal mask
        mask |= (cond.astype(np.uint16) * value)

    # Convert boolean pixel mask to boolean per-dot arrays at positions
    mb = mask_bool
    # Map according to Braille dot numbering
    bit(mb[0::4, 0::2], 0x01)  # dot 1
    bit(mb[1::4, 0::2], 0x02)  # dot 2
    bit(mb[2::4, 0::2], 0x04)  # dot 3
    bit(mb[0::4, 1::2], 0x08)  # dot 4 (right top)
    bit(mb[1::4, 1::2], 0x10)  # dot 5
    bit(mb[2::4, 1::2], 0x20)  # dot 6
    bit(mb[3::4, 0::2], 0x40)  # dot 7
    bit(mb[3::4, 1::2], 0x80)  # dot 8

    codepoints = 0x2800 + mask
    # Turn into lines
    lines: list[str] = []
    for row in codepoints:
        # Using Python-level join with chr is fast enough here
        lines.append("".join(map(chr, row)))
    return lines


def frame_to_braille_color(
    frame_bgr: np.ndarray,
    width_chars: int,
    threshold: int = 130,
    invert: bool = False,
    aspect: float = 0.5,
    adaptive: str = "none",
    block_size: int = 15,
    adaptive_c: int = 5,
    gamma: float = 1.0,
    clahe_clip: Optional[float] = None,
    clahe_grid: int = 8,
    blur_kernel: int = 0,
    edge_boost: bool = False,
    canny_low: int = 100,
    canny_high: int = 200,
) -> list[str]:
    """Convert a BGR frame to color Braille using 24-bit ANSI color per cell."""
    h, w = frame_bgr.shape[:2]
    Wc, Hc = compute_target_cell_size(w, h, width_chars, aspect=aspect)
    target_w = max(2, Wc * 2)
    target_h = max(4, Hc * 4)
    small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    # Compute grayscale for dot pattern
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    H = (gray.shape[0] // 4) * 4
    W = (gray.shape[1] // 2) * 2
    if H == 0 or W == 0:
        return []
    g = gray[:H, :W]
    color = small[:H, :W, ::-1]  # convert BGR->RGB by reversing last axis when emitting

    cells_h = H // 4
    cells_w = W // 2

    # Build boolean mask as in grayscale path
    mask_bool = _build_mask(
        g,
        threshold=threshold,
        invert=invert,
        adaptive=adaptive,
        block_size=block_size,
        adaptive_c=adaptive_c,
        gamma=gamma,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        blur_kernel=blur_kernel,
        edge_boost=edge_boost,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    mask = np.zeros((cells_h, cells_w), dtype=np.uint16)
    mb = mask_bool
    mask |= mb[0::4, 0::2].astype(np.uint16) * 0x01
    mask |= mb[1::4, 0::2].astype(np.uint16) * 0x02
    mask |= mb[2::4, 0::2].astype(np.uint16) * 0x04
    mask |= mb[0::4, 1::2].astype(np.uint16) * 0x08
    mask |= mb[1::4, 1::2].astype(np.uint16) * 0x10
    mask |= mb[2::4, 1::2].astype(np.uint16) * 0x20
    mask |= mb[3::4, 0::2].astype(np.uint16) * 0x40
    mask |= mb[3::4, 1::2].astype(np.uint16) * 0x80
    codepoints = (0x2800 + mask).astype(np.uint16)

    # Average RGB per cell using reshape trick: (Hc,4,Wc,2,3) -> mean over (1,3)
    rgb = color.reshape(cells_h, 4, cells_w, 2, 3).mean(axis=(1, 3)).astype(np.uint8)

    lines: list[str] = []
    # Emit per-cell color (foreground) + glyph. Reset at end of each line.
    for y in range(cells_h):
        parts = []
        row_codes = codepoints[y]
        row_rgb = rgb[y]
        for x in range(cells_w):
            r, g_, b = row_rgb[x]
            parts.append(f"\x1b[38;2;{r};{g_};{b}m" + chr(int(row_codes[x])))
        parts.append("\x1b[0m")
        lines.append("".join(parts))
    return lines


def _open_capture(path: str, backend: str) -> Any:
    try:
        if backend == "ffmpeg" and hasattr(cv2, "CAP_FFMPEG"):
            return cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if backend == "msmf" and hasattr(cv2, "CAP_MSMF"):
            return cv2.VideoCapture(path, cv2.CAP_MSMF)
        return cv2.VideoCapture(path)
    except Exception:
        return cv2.VideoCapture(path)


def _get_positions(cap: Any) -> Tuple[float, int]:
    # Returns (msec, frame_idx) best-effort
    try:
        msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    except Exception:
        msec = 0.0
    try:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    except Exception:
        frame_idx = 0
    return msec, frame_idx


def play_video_braille(
    path: str,
    width: int = 120,
    mode: str = "braille",
    threshold: int = 130,
    invert: bool = False,
    fps_limit: Optional[float] = None,
    aspect: float = 0.5,
    adaptive: str = "none",
    block_size: int = 15,
    adaptive_c: int = 5,
    gamma: float = 1.0,
    clahe_clip: Optional[float] = None,
    clahe_grid: int = 8,
    blur_kernel: int = 0,
    edge_boost: bool = False,
    canny_low: int = 100,
    canny_high: int = 200,
    sample_seconds: Optional[float] = None,
    sample_output: Optional[str] = None,
    sample_step: int = 3,
    sample_ansi: bool = False,
    no_display: bool = False,
    log_file: Optional[str] = None,
    seek: Optional[float] = None,
    audio: bool = False,
    audio_volume: float = 1.0,
    audio_player: Optional[str] = None,
    audio_resync: bool = False,  # If True, re-sync audio on restarts/skips (may cause stutter)
    sync_to_audio: bool = True,  # If True, disable FPS limit when audio is enabled
    debug: bool = False,
    on_error: str = "continue",  # or "stop"
    max_bad_reads: int = 60,
    backend: str = "auto",  # auto|ffmpeg|msmf|any
    checkpoint_secs: Optional[float] = None,
    restart_on_stall: bool = True,
    max_restarts: int = 50,  # Much higher restart limit
    restart_seek_back: float = 1.0,  # Go back further on restart
    stall_skip_forward: float = 2.0,  # If max restarts hit, skip forward N seconds
):
    if cv2 is None:
        print("OpenCV (opencv-python) is required for this script.\nInstall: pip install opencv-python")
        sys.exit(1)

    # Choose backend when possible
    cap = _open_capture(path, backend)
    if not cap.isOpened():
        print(f"Failed to open video: {path}")
        sys.exit(1)

    # Optional seek to start time (seconds)
    if seek is not None and seek > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(seek) * 1000.0)

    # Reduce buffering if backend supports it (helps latency/stability)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0 or src_fps != src_fps:  # NaN check
        src_fps = 24.0

    target_fps = min(src_fps, fps_limit) if fps_limit else src_fps
    frame_interval = 1.0 / max(1e-6, target_fps)

    # Enhanced timing for audio sync
    video_start_time = time.perf_counter()
    expected_video_time = 0.0  # Expected video position in seconds
    audio_start_offset = seek or 0.0  # Track audio offset for sync
    sync_check_interval = 60  # Check sync every N frames
    max_drift_threshold = 0.5  # Max drift in seconds before correction
    try:
        total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:
        total_frames_reported = -1

    enable_windows_utf8()
    init_ansi()
    hide_cursor()
    # Clear screen and home only if displaying
    if not no_display:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()

    last_lines_count = None
    t0 = time.perf_counter()
    next_time = t0
    frames = 0
    first_ts = None
    sample_fh = None
    bad_reads = 0
    events: list[dict] = []

    # Auto-disable FPS limit when audio is enabled for sync
    if audio and sync_to_audio and fps_limit is not None:
        events.append({"t": 0.0, "type": "sync_auto_fps", "disabled_fps_limit": fps_limit})
        target_fps = src_fps  # Use source FPS when auto-syncing
        frame_interval = 1.0 / max(1e-6, target_fps)

    # Optional audio via ffplay
    audio_proc = None
    if audio:
        player = audio_player or shutil.which("ffplay")
        if not player:
            print("Warning: ffplay (from FFmpeg) not found; audio disabled.")
        else:
            cmd = [player, "-nodisp", "-autoexit", "-loglevel", "error"]
            if seek is not None and seek > 0:
                cmd.extend(["-ss", str(float(seek))])
            if audio_volume != 1.0:
                cmd.extend(["-af", f"volume={audio_volume}"])
            cmd.extend(["-i", path])
            creationflags = 0
            preexec_fn = None
            if os.name == "nt":
                # Don't create new process group - keep audio in same session
                creationflags = 0
            else:
                preexec_fn = os.setsid
            try:
                audio_proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                    preexec_fn=preexec_fn,
                )
            except Exception as e:
                print(f"Warning: could not start audio player: {e}")
                audio_proc = None
    last_checkpoint = time.perf_counter()
    restarts = 0
    last_msec = 0.0
    # Stall detection
    stall_frame_count = 0
    last_frame_hash = None
    max_stall_frames = 30  # Restart after 30 identical frames
    position_stall_count = 0
    last_position_msec = 0.0
    # Consider position unchanged if progress < epsilon_ms
    position_epsilon_ms = 5.0
    max_position_stalls = 20  # Restart after ~2s at 10fps

    # Frame-index stall detection (some backends update frames but not msec or vice versa)
    last_frame_index = -1
    frame_index_stall_count = 0
    max_frame_index_stalls = 25

    # Visual low-change stall detection (frames changing too little)
    prev_small_gray = None
    low_change_count = 0
    max_low_change_frames = 30  # ~3s at 10fps
    low_change_threshold = 0.3  # mean absolute difference threshold (0-255)
    # Backend failover setup
    backend_options = ["ffmpeg", "msmf", "dshow", "any"]
    backend_idx = backend_options.index(backend) if backend in backend_options else 0
    backend_failover_count = 0
    # Metrics aggregation (simple heuristics for clarity)
    metrics_n = 0
    metrics_sum_nonblank = 0.0
    metrics_sum_dots = 0.0
    metrics_sum_trans = 0.0
    end_reason = None

    # Helper: attempt to restart capture cleanly on stalls
    def attempt_restart(reason: str) -> bool:
        nonlocal cap, restarts, bad_reads, next_time, last_position_msec, stall_frame_count, last_frame_hash, audio_proc
        nonlocal backend_idx, backend, backend_failover_count
        try:
            if not restart_on_stall:
                return False
            # If we've exhausted restarts, try skip-forward instead of giving up
            if restarts >= max_restarts:
                # Backend failover: try next backend if available
                if backend_failover_count < len(backend_options):
                    backend_idx = (backend_idx + 1) % len(backend_options)
                    backend = backend_options[backend_idx]
                    backend_failover_count += 1
                    events.append({
                        "t": round(time.perf_counter() - t0, 3),
                        "type": "backend_failover",
                        "new_backend": backend,
                        "failover_count": backend_failover_count
                    })
                    try:
                        cap.release()
                    except Exception:
                        pass
                    new_cap = _open_capture(path, backend)
                    if not new_cap.isOpened():
                        return False
                    cap = new_cap
                    restarts = 0
                    bad_reads = 0
                    stall_frame_count = 0
                    last_frame_hash = None
                    last_position_msec = 0.0
                    next_time = time.perf_counter()
                    return True
                try:
                    skip_ms = max(0.0, float(stall_skip_forward)) * 1000.0
                except Exception:
                    skip_ms = 0.0
                new_msec = max(0.0, (last_msec or 0.0) + skip_ms)
                events.append({
                    "t": round(time.perf_counter() - t0, 3),
                    "type": "stall_skip_forward",
                    "reason": reason,
                    "skip_ms": round(skip_ms, 2),
                    "new_msec": round(new_msec, 2),
                })
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, new_msec)
                except Exception:
                    pass
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                # Re-sync audio to new position if enabled (opt-in)
                if audio_resync and audio and (audio_player or shutil.which("ffplay")):
                    try:
                        # Stop current audio
                        if audio_proc is not None:
                            if os.name == "nt":
                                audio_proc.terminate()
                            else:
                                try:
                                    os.killpg(os.getpgid(audio_proc.pid), signal.SIGTERM)
                                except Exception:
                                    audio_proc.terminate()
                            try:
                                audio_proc.wait(timeout=2)
                            except Exception:
                                try:
                                    audio_proc.kill()
                                    audio_proc.wait(timeout=1)
                                except Exception:
                                    pass
                        # Start new audio at new position
                        player = audio_player or shutil.which("ffplay")
                        if player:
                            cmd = [player, "-nodisp", "-autoexit", "-loglevel", "error", "-ss", str(new_msec / 1000.0), "-i", path]
                            if audio_volume != 1.0:
                                cmd = [player, "-nodisp", "-autoexit", "-loglevel", "error", "-ss", str(new_msec / 1000.0), "-af", f"volume={audio_volume}", "-i", path]
                            creationflags = 0
                            preexec_fn = None
                            if os.name != "nt":
                                preexec_fn = os.setsid
                            audio_proc = subprocess.Popen(
                                cmd,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=creationflags,
                                preexec_fn=preexec_fn,
                            )
                            events.append({
                                "t": round(time.perf_counter() - t0, 3),
                                "type": "audio_resync",
                                "sec": round(new_msec / 1000.0, 3),
                            })
                    except Exception as _e:
                        events.append({"t": round(time.perf_counter() - t0, 3), "type": "audio_resync_failed", "msg": str(_e)})
                bad_reads = 0
                stall_frame_count = 0
                last_frame_hash = None
                last_position_msec = new_msec
                next_time = time.perf_counter()
                return True
            restarts += 1
            # Convert seek-back seconds to milliseconds for CAP_PROP_POS_MSEC
            resume_msec = max(0.0, (last_msec or 0.0) - max(0.0, float(restart_seek_back)) * 1000.0)
            events.append({
                "t": round(time.perf_counter() - t0, 3),
                "type": "restart_capture",
                "reason": reason,
                "restarts": restarts,
                "resume_msec": round(resume_msec, 2),
            })
            try:
                cap.release()
            except Exception:
                pass
            new_cap = _open_capture(path, backend)
            if not new_cap.isOpened():
                return False
            cap = new_cap
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, resume_msec)
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # Re-sync audio to resume position if enabled (opt-in)
            if audio_resync and audio and (audio_player or shutil.which("ffplay")):
                try:
                    if audio_proc is not None:
                        if os.name == "nt":
                            audio_proc.terminate()
                        else:
                            try:
                                os.killpg(os.getpgid(audio_proc.pid), signal.SIGTERM)
                            except Exception:
                                audio_proc.terminate()
                        try:
                            audio_proc.wait(timeout=2)
                        except Exception:
                            try:
                                audio_proc.kill()
                                audio_proc.wait(timeout=1)
                            except Exception:
                                pass
                    player = audio_player or shutil.which("ffplay")
                    if player:
                        cmd = [player, "-nodisp", "-autoexit", "-loglevel", "error", "-ss", str(resume_msec / 1000.0), "-i", path]
                        if audio_volume != 1.0:
                            cmd = [player, "-nodisp", "-autoexit", "-loglevel", "error", "-ss", str(resume_msec / 1000.0), "-af", f"volume={audio_volume}", "-i", path]
                        creationflags = 0
                        preexec_fn = None
                        if os.name != "nt":
                            preexec_fn = os.setsid
                        audio_proc = subprocess.Popen(
                            cmd,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=creationflags,
                            preexec_fn=preexec_fn,
                        )
                        events.append({
                            "t": round(time.perf_counter() - t0, 3),
                            "type": "audio_resync",
                            "sec": round(resume_msec / 1000.0, 3),
                        })
                except Exception as _e:
                    events.append({"t": round(time.perf_counter() - t0, 3), "type": "audio_resync_failed", "msg": str(_e)})
            bad_reads = 0
            stall_frame_count = 0
            last_frame_hash = None
            last_position_msec = 0.0
            next_time = time.perf_counter()  # resync pacing
            return True
        except Exception as e:
            events.append({"t": round(time.perf_counter() - t0, 3), "type": "restart_failed", "msg": str(e), "reason": reason})
            return False

    def analyze_lines_for_metrics(lines: list[str], color_mode: bool) -> tuple[float, float, float]:
        # Returns (nonblank_ratio, dots_density, transitions_per_char)
        if color_mode:
            lines_plain = [strip_ansi(L) for L in lines]
        else:
            lines_plain = lines
        total = 0
        nonblank = 0
        dots_total = 0
        transitions = 0
        for s in lines_plain:
            prev = None
            for ch in s:
                cp = ord(ch)
                if 0x2800 <= cp <= 0x28FF:
                    val = cp - 0x2800
                    total += 1
                    nb = 1 if val != 0 else 0
                    nonblank += nb
                    dots_total += int(val).bit_count()
                    if prev is not None:
                        transitions += 1 if (prev != nb) else 0
                    prev = nb
        if total == 0:
            return 0.0, 0.0, 0.0
        nonblank_ratio = nonblank / total
        dots_density = dots_total / (total * 8.0)
        # Approximate per-character horizontal transition rate
        transitions_per_char = transitions / max(1, total)
        return nonblank_ratio, dots_density, transitions_per_char

    def strip_ansi(s: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", s)

    if sample_output:
        try:
            sample_fh = open(sample_output, "w", encoding="utf-8")
            meta = {
                "started": datetime.now().isoformat(timespec="seconds"),
                "video": path,
                "mode": mode,
                "width": width,
                "fps_limit": fps_limit,
                "aspect": aspect,
                "adaptive": adaptive,
                "block_size": block_size,
                "adaptive_c": adaptive_c,
                "gamma": gamma,
                "clahe": clahe_clip,
                "clahe_grid": clahe_grid,
                "blur_kernel": blur_kernel,
                "edge_boost": edge_boost,
                "canny_low": canny_low,
                "canny_high": canny_high,
            }
            sample_fh.write(f"# video_braille sample\n# meta: {json.dumps(meta)}\n\n")
        except Exception as e:
            print(f"Could not open sample file: {e}")
            sample_fh = None
    try:
        while True:
            ret = False
            frame = None
            try:
                ret, frame = cap.read()
            except Exception as e:
                events.append({"t": round(time.perf_counter() - t0, 3), "type": "read_exception", "msg": str(e)})
                ret = False
                frame = None
            if not ret or frame is None:
                bad_reads += 1
                events.append({"t": round(time.perf_counter() - t0, 3), "type": "bad_read", "count": bad_reads})
                if on_error == "continue" and bad_reads <= max_bad_reads:
                    # try to advance one frame and continue
                    try:
                        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 1)
                    except Exception:
                        pass
                    time.sleep(0.005)
                    continue
                # Past threshold: attempt restart?
                if on_error == "continue" and restart_on_stall and restarts < max_restarts:
                    try:
                        restarts += 1
                        # determine resume position slightly earlier (seconds -> milliseconds)
                        resume_msec = max(0.0, (last_msec or 0.0) - max(0.0, float(restart_seek_back)) * 1000.0)
                        events.append({
                            "t": round(time.perf_counter() - t0, 3),
                            "type": "restart_capture",
                            "restarts": restarts,
                            "resume_msec": round(resume_msec, 2),
                        })
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = _open_capture(path, backend)
                        if not cap.isOpened():
                            end_reason = "restart_open_failed"
                            break
                        # apply seek
                        try:
                            cap.set(cv2.CAP_PROP_POS_MSEC, resume_msec)
                        except Exception:
                            pass
                        # reduce buffering again
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        bad_reads = 0
                        next_time = time.perf_counter()  # resync pacing
                        continue
                    except Exception as e:
                        events.append({"t": round(time.perf_counter() - t0, 3), "type": "restart_failed", "msg": str(e)})
                        end_reason = "restart_exception"
                        break
                end_reason = "bad_reads_exceeded"
                break
            bad_reads = 0

            if mode == "color-braille":
                lines = frame_to_braille_color(
                    frame,
                    width_chars=width,
                    threshold=threshold,
                    invert=invert,
                    aspect=aspect,
                    adaptive=adaptive,
                    block_size=block_size,
                    adaptive_c=adaptive_c,
                    gamma=gamma,
                    clahe_clip=clahe_clip,
                    clahe_grid=clahe_grid,
                    blur_kernel=blur_kernel,
                    edge_boost=edge_boost,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
            else:
                lines = frame_to_braille_gray(
                    frame,
                    width_chars=width,
                    threshold=threshold,
                    invert=invert,
                    aspect=aspect,
                    adaptive=adaptive,
                    block_size=block_size,
                    adaptive_c=adaptive_c,
                    gamma=gamma,
                    clahe_clip=clahe_clip,
                    clahe_grid=clahe_grid,
                    blur_kernel=blur_kernel,
                    edge_boost=edge_boost,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )

            # Track last known timestamp and frame index for restart positioning
            try:
                last_msec, curr_frame_idx = _get_positions(cap)
            except Exception:
                curr_frame_idx = -1

            # Stall detection - check if we're getting identical frames
            frame_hash = hash(frame.tobytes()) if frame is not None else None
            if frame_hash == last_frame_hash:
                stall_frame_count += 1
                if stall_frame_count >= max_stall_frames:
                    events.append({"t": round(time.perf_counter() - t0, 3), "type": "stall_detected", "identical_frames": stall_frame_count})
                    # Actively restart the capture on stall
                    if attempt_restart("frame_stall"):
                        continue
                    else:
                        # Fallback: force termination via bad_reads path
                        bad_reads = max_bad_reads + 1
                        stall_frame_count = 0
            else:
                stall_frame_count = 0
                last_frame_hash = frame_hash

            # Position-based stall detection
            if abs(last_msec - last_position_msec) < position_epsilon_ms:  # Same position (within epsilon)
                position_stall_count += 1
                if position_stall_count >= max_position_stalls:
                    events.append({"t": round(time.perf_counter() - t0, 3), "type": "position_stall_detected", "stall_count": position_stall_count})
                    # Actively restart the capture on position stall
                    if attempt_restart("position_stall"):
                        continue
                    else:
                        bad_reads = max_bad_reads + 1
                        position_stall_count = 0
            else:
                position_stall_count = 0
                last_position_msec = last_msec

            # Frame-index stall detection
            if curr_frame_idx <= last_frame_index:
                frame_index_stall_count += 1
                if frame_index_stall_count >= max_frame_index_stalls:
                    events.append({"t": round(time.perf_counter() - t0, 3), "type": "frame_index_stall", "stall_count": frame_index_stall_count})
                    if attempt_restart("frame_index_stall"):
                        continue
                    else:
                        bad_reads = max_bad_reads + 1
                        frame_index_stall_count = 0
            else:
                frame_index_stall_count = 0
                last_frame_index = curr_frame_idx

            # Visual low-change stall detection
            try:
                sg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sg = cv2.resize(sg, (64, 36), interpolation=cv2.INTER_AREA)
                if prev_small_gray is not None:
                    mad = float(np.mean(np.abs(sg.astype(np.int16) - prev_small_gray.astype(np.int16))))
                    if mad < low_change_threshold:
                        low_change_count += 1
                        if low_change_count >= max_low_change_frames:
                            events.append({"t": round(time.perf_counter() - t0, 3), "type": "low_change_stall", "mad": round(mad,3), "threshold": low_change_threshold, "backend": backend_options[backend_idx]})
                            if attempt_restart("low_change_stall"):
                                prev_small_gray = None
                                continue
                            else:
                                bad_reads = max_bad_reads + 1
                                low_change_count = 0
                    else:
                        low_change_count = 0
                prev_small_gray = sg
            except Exception:
                # If visual check fails, ignore
                pass

            # Enhanced frame pacing with audio sync
            now = time.perf_counter()

            if audio and sync_to_audio:
                # Calculate expected video position based on frames processed
                expected_video_time = frames / target_fps
                actual_elapsed = now - video_start_time

                # Calculate how far ahead/behind we are
                time_drift = actual_elapsed - expected_video_time

                # Periodic sync check and correction
                if frames % sync_check_interval == 0 and frames > 0:
                    video_position_seconds = expected_video_time + audio_start_offset

                    # Check if drift exceeds threshold
                    if abs(time_drift) > max_drift_threshold:
                        events.append({
                            "t": round(now - t0, 3),
                            "type": "large_drift_detected",
                            "drift_ms": round(time_drift * 1000, 1),
                            "video_pos": round(video_position_seconds, 3),
                            "correction_needed": True
                        })

                        # If we're significantly behind, skip some video frames
                        if time_drift > max_drift_threshold:
                            skip_frames = int(time_drift * target_fps)
                            try:
                                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos + skip_frames)
                                events.append({
                                    "t": round(now - t0, 3),
                                    "type": "video_skip_correction",
                                    "skipped_frames": skip_frames,
                                    "drift_ms": round(time_drift * 1000, 1)
                                })
                                # Reset timing after skip
                                video_start_time = now - expected_video_time
                            except Exception:
                                pass

                # Only sleep if we're ahead of schedule
                if time_drift < 0:
                    sleep_time = -time_drift
                    time.sleep(sleep_time)

                # Log significant drift for debugging
                if debug and abs(time_drift) > 0.1:
                    events.append({
                        "t": round(now - t0, 3),
                        "type": "sync_drift",
                        "drift_ms": round(time_drift * 1000, 1),
                        "expected_pos": round(expected_video_time, 3),
                        "actual_elapsed": round(actual_elapsed, 3)
                    })
            else:
                # Original frame pacing for non-audio playback
                if now < next_time:
                    time.sleep(max(0.0, next_time - now))
                next_time += frame_interval

            # Display
            if not no_display:
                try:
                    frame_str = ""
                    if last_lines_count is None:
                        frame_str += "\x1b[2J\x1b[H"  # first clear
                    else:
                        home_and_clear_lines(0)
                    for L in lines:
                        try:
                            frame_str += L + "\n"
                        except UnicodeEncodeError:
                            safe_line = L.encode('utf-8', errors='replace').decode('utf-8')
                            frame_str += safe_line + "\n"
                    sys.stdout.write(frame_str)
                    sys.stdout.flush()
                except (BrokenPipeError, OSError, UnicodeEncodeError):
                    # Output pipe closed; stop rendering
                    no_display = True
            last_lines_count = len(lines)
            frames += 1

            # Sampling to file
            if sample_fh and (frames % max(1, sample_step) == 0):
                if mode == "color-braille" and not sample_ansi:
                    out_lines = [strip_ansi(L) for L in lines]
                else:
                    out_lines = lines
                sample_fh.write(f"\n--- FRAME {frames} t={now - t0:.2f}s ---\n")
                for L in out_lines:
                    sample_fh.write(L + "\n")

            # Metrics sampling (cheap heuristic)
            if (log_file or sample_output) and (frames % 6 == 0):
                nb, dd, tr = analyze_lines_for_metrics(lines, color_mode=(mode == "color-braille"))
                metrics_n += 1
                metrics_sum_nonblank += nb
                metrics_sum_dots += dd
                metrics_sum_trans += tr
                if debug:
                    events.append({"t": round(time.perf_counter() - t0, 3), "type": "metrics", "nb": round(nb,4), "dd": round(dd,4), "tr": round(tr,4)})

            # Removed early stop after N seconds (sample_seconds) logic

            # Periodic checkpoint logging for crash forensics
            if checkpoint_secs and log_file and (time.perf_counter() - last_checkpoint) >= checkpoint_secs:
                try:
                    elapsed = max(1e-6, time.perf_counter() - t0)
                    achieved_fps = frames / elapsed
                    ck = {
                        "ended": datetime.now().isoformat(timespec="seconds"),
                        "video": path,
                        "mode": mode,
                        "width": width,
                        "fps_limit": fps_limit,
                        "achieved_fps": round(achieved_fps, 3),
                        "frames": frames,
                        "aspect": aspect,
                        "adaptive": adaptive,
                        "block_size": block_size,
                        "adaptive_c": adaptive_c,
                        "gamma": gamma,
                        "clahe": clahe_clip,
                        "clahe_grid": clahe_grid,
                        "blur_kernel": blur_kernel,
                        "edge_boost": edge_boost,
                        "canny_low": canny_low,
                        "canny_high": canny_high,
                        "checkpoint": True,
                    }
                    tmp = f"{log_file}.tmp"
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(ck, fh, ensure_ascii=False, indent=2)
                    try:
                        os.replace(tmp, log_file)
                    except Exception:
                        pass
                except Exception:
                    pass
                last_checkpoint = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        cap.release()
        if sample_fh:
            sample_fh.close()

        # Stop audio if running - CRITICAL: Must be in finally block
        if audio_proc is not None:
            try:
                if os.name == "nt":
                    # For Windows: direct terminate is more reliable than signals
                    audio_proc.terminate()
                else:
                    try:
                        os.killpg(os.getpgid(audio_proc.pid), signal.SIGTERM)
                    except Exception:
                        audio_proc.terminate()
            except Exception:
                pass
            try:
                audio_proc.wait(timeout=3)
            except Exception:
                try:
                    audio_proc.kill()
                    audio_proc.wait(timeout=1)
                except Exception:
                    pass

    # Optional JSON log
    # Ensure an end_reason is set for clarity
    if not end_reason:
        end_reason = "loop_exit"
    if log_file:
        try:
            elapsed = max(1e-6, time.perf_counter() - t0)
            achieved_fps = frames / elapsed
            log = {
                "ended": datetime.now().isoformat(timespec="seconds"),
                "video": path,
                "mode": mode,
                "width": width,
                "fps_limit": fps_limit,
                "achieved_fps": round(achieved_fps, 3),
                "frames": frames,
                "src_fps": round(src_fps, 4),
                "target_fps": round(target_fps, 4),
                "reported_total_frames": total_frames_reported,
                "aspect": aspect,
                "adaptive": adaptive,
                "block_size": block_size,
                "adaptive_c": adaptive_c,
                "gamma": gamma,
                "clahe": clahe_clip,
                "clahe_grid": clahe_grid,
                "blur_kernel": blur_kernel,
                "edge_boost": edge_boost,
                "canny_low": canny_low,
                "canny_high": canny_high,
                "end_reason": end_reason,
                "last_msec": last_msec,
                "last_frame_index": last_frame_index if 'last_frame_index' in locals() else None,
                "restarts": restarts,
            }
            if metrics_n > 0:
                log.update({
                    "metric_nonblank_ratio": round(metrics_sum_nonblank / metrics_n, 4),
                    "metric_dots_density": round(metrics_sum_dots / metrics_n, 4),
                    "metric_transitions_per_char": round(metrics_sum_trans / metrics_n, 4),
                })
            if events:
                log["events"] = events[-200:]  # keep last 200 events
            with open(log_file, "w", encoding="utf-8") as fh:
                json.dump(log, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Could not write log file: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Play a video as live (color) Braille in the terminal")
    p.add_argument("video", help="Path to input video (mp4/mkv/avi…)")
    p.add_argument("--width", type=int, default=120, help="Target character width (default: 120)")
    p.add_argument("--mode", choices=["braille", "color-braille"], default="braille",
                   help="Rendering mode")
    p.add_argument("--threshold", type=int, default=130, help="Threshold for dot activation (0-255); used when --adaptive=none")
    p.add_argument("--invert", action="store_true", help="Invert brightness mapping")
    p.add_argument("--fps-limit", type=float, default=None, help="Limit playback FPS (default: source FPS)")
    p.add_argument("--aspect", type=float, default=0.5, help="Vertical compensation factor (terminal char aspect); try 0.45–0.6")
    p.add_argument("--adaptive", choices=["none", "mean", "gaussian", "otsu"], default="none",
                   help="Adaptive/local thresholding mode for clearer edges")
    p.add_argument("--block-size", type=int, default=15, help="Adaptive threshold block size (odd >=3)")
    p.add_argument("--adaptive-c", type=int, default=5, help="Adaptive threshold constant C")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma correction; <1 brightens, >1 darkens")
    p.add_argument("--clahe", type=float, default=None, metavar="CLIP", help="Enable CLAHE with clip limit (e.g., 2.0)")
    p.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size (e.g., 8)")
    p.add_argument("--blur-kernel", type=int, default=0, help="Gaussian blur kernel size (odd), reduces noise before threshold")
    p.add_argument("--edge-boost", action="store_true", help="Boost edges via Canny and OR into dots")
    p.add_argument("--canny-low", type=int, default=100, help="Canny low threshold")
    p.add_argument("--canny-high", type=int, default=200, help="Canny high threshold")
    p.add_argument("--preset", choices=["none", "clarity", "color", "fast"], default="none",
                   help="Apply a tuned preset of options")
    p.add_argument("--sample-seconds", type=float, default=None, help="Play only this many seconds and then stop")
    p.add_argument("--sample-output", type=str, default=None, help="Write sampled frames to this text file")
    p.add_argument("--sample-step", type=int, default=3, help="Record every Nth frame to the sample file")
    p.add_argument("--sample-ansi", action="store_true", help="Keep ANSI color codes in sample output (color mode)")
    p.add_argument("--no-display", action="store_true", help="Do not render to terminal (sampling/logging only)")
    p.add_argument("--log-file", type=str, default=None, help="Write a JSON summary of parameters and performance")
    p.add_argument("--seek", type=float, default=None, help="Start playback at this timestamp (seconds)")
    p.add_argument("--audio", action="store_true", help="Play audio via ffplay (requires FFmpeg)")
    p.add_argument("--audio-volume", type=float, default=1.0, help="Audio volume scalar for ffplay (e.g., 0.8, 1.2)")
    p.add_argument("--audio-player", type=str, default=None, help="Path to ffplay if not on PATH")
    p.add_argument("--audio-resync", action="store_true", help="Re-sync audio on restarts/skips (may cause audible stutter)")
    p.add_argument("--sync-to-audio", action="store_true", default=True, help="Automatically disable FPS limit when audio is enabled for better sync")
    p.add_argument("--no-sync-to-audio", action="store_true", help="Disable automatic audio sync optimizations")
    p.add_argument("--debug", action="store_true", help="Add periodic metrics/events to log file")
    p.add_argument("--on-error", choices=["continue","stop"], default="continue", help="Behavior on decode errors")
    p.add_argument("--max-bad-reads", type=int, default=60, help="Max consecutive failed frame reads before stopping")
    p.add_argument("--backend", choices=["auto","ffmpeg","msmf","any"], default="auto", help="OpenCV capture backend")
    p.add_argument("--checkpoint-secs", type=float, default=None, help="Write partial log every N seconds")
    p.add_argument("--restart-on-stall", action="store_true", help="Automatically reopen the video if reads stall")
    p.add_argument("--max-restarts", type=int, default=50, help="Max automatic reopen attempts on stall")
    p.add_argument("--restart-seek-back", type=float, default=1.0, help="How many seconds to seek back on restart")
    p.add_argument("--stall-skip-forward", type=float, default=2.0, help="If stalls persist past max restarts, skip forward N seconds")
    # Auto-tune
    p.add_argument("--auto-tune", choices=["none", "color", "braille"], default="none",
                   help="Try a small set of tuned variants and produce a ranked summary")
    p.add_argument("--tune-seconds", type=float, default=12.0, help="Per-variant duration in seconds")
    p.add_argument("--tune-prefix", type=str, default="tune", help="Prefix for outputs: <prefix>_<id>.txt/.json and summary file")
    return p.parse_args()


def main():
    args = parse_args()
    # Apply presets (can be overridden by explicit flags)
    if args.preset != "none":
        if args.preset == "clarity":
            args.mode = "braille"
            args.width = max(args.width, 140)
            args.fps_limit = args.fps_limit or 12
            args.aspect = 0.53
            args.clahe = 2.0 if args.clahe is None else args.clahe
            args.adaptive = "gaussian" if args.adaptive == "none" else args.adaptive
            args.block_size = max(args.block_size, 21)
            args.adaptive_c = max(args.adaptive_c, 5)
            args.gamma = 0.9 if abs(args.gamma - 1.0) < 1e-6 else args.gamma
            args.blur_kernel = 3 if args.blur_kernel == 0 else args.blur_kernel
            args.edge_boost = True or args.edge_boost
            args.canny_low = min(args.canny_low, 90)
            args.canny_high = max(args.canny_high, 200)
        elif args.preset == "color":
            args.mode = "color-braille"
            args.width = max(args.width, 110)
            args.fps_limit = args.fps_limit or 10
            args.aspect = 0.53
            args.clahe = 2.0 if args.clahe is None else args.clahe
            args.adaptive = "gaussian" if args.adaptive == "none" else args.adaptive
            args.block_size = max(args.block_size, 21)
            args.gamma = 0.95 if abs(args.gamma - 1.0) < 1e-6 else args.gamma
            args.blur_kernel = 3 if args.blur_kernel == 0 else args.blur_kernel
            args.edge_boost = True or args.edge_boost
        elif args.preset == "fast":
            args.adaptive = "none"
            args.edge_boost = False
            args.clahe = None
            args.gamma = 1.0
            args.blur_kernel = 0
    if not os.path.exists(args.video):
        print(f"Input not found: {args.video}")
        sys.exit(1)

    # Auto-tune pipeline
    if args.auto_tune != "none":
        cases = []
        if args.auto_tune == "color":
            base = dict(mode="color-braille", aspect=0.56, adaptive="gaussian")
            cases = [
                dict(label="C1_clean", width=120, clahe=1.6, gamma=0.9, block_size=27, adaptive_c=3, blur_kernel=3, edge_boost=False),
                dict(label="C2_clean_wide", width=130, clahe=1.6, gamma=0.9, block_size=29, adaptive_c=4, blur_kernel=3, edge_boost=False),
                dict(label="C3_light_edges", width=120, clahe=1.8, gamma=0.92, block_size=27, adaptive_c=4, blur_kernel=5, edge_boost=True, canny_low=120, canny_high=240),
                dict(label="C4_local_contrast", width=120, clahe=2.2, gamma=0.92, block_size=31, adaptive_c=6, blur_kernel=3, edge_boost=False),
                dict(label="C5_soft", width=115, clahe=1.4, gamma=0.95, block_size=25, adaptive_c=2, blur_kernel=3, edge_boost=False),
                dict(label="C6_otsu_mix", width=120, clahe=1.6, gamma=0.9, adaptive="otsu", blur_kernel=3, edge_boost=False),
            ]
        else:  # braille
            base = dict(mode="braille", aspect=0.56)
            cases = [
                dict(label="G1_baseline", width=140, clahe=2.0, gamma=0.9, adaptive="gaussian", block_size=27, adaptive_c=3, blur_kernel=3, edge_boost=False),
                dict(label="G2_edges_light", width=140, clahe=1.8, gamma=0.95, adaptive="gaussian", block_size=27, adaptive_c=4, blur_kernel=3, edge_boost=True, canny_low=120, canny_high=240),
                dict(label="G3_otsu", width=140, clahe=1.6, gamma=0.9, adaptive="otsu", blur_kernel=3, edge_boost=False),
            ]

        results = []
        for idx, spec in enumerate(cases, start=1):
            params = dict(
                path=args.video,
                width=spec.get("width", args.width),
                mode=spec.get("mode", base.get("mode")),
                threshold=args.threshold,
                invert=args.invert,
                fps_limit=args.fps_limit or (10.0 if base.get("mode") == "color-braille" else 12.0),
                aspect=spec.get("aspect", base.get("aspect", args.aspect)),
                adaptive=spec.get("adaptive", base.get("adaptive", args.adaptive)),
                block_size=spec.get("block_size", args.block_size),
                adaptive_c=spec.get("adaptive_c", args.adaptive_c),
                gamma=spec.get("gamma", args.gamma),
                clahe_clip=spec.get("clahe", args.clahe),
                clahe_grid=args.clahe_grid,
                blur_kernel=spec.get("blur_kernel", args.blur_kernel),
                edge_boost=spec.get("edge_boost", args.edge_boost),
                canny_low=spec.get("canny_low", args.canny_low),
                canny_high=spec.get("canny_high", args.canny_high),
                sample_seconds=float(args.tune_seconds),
                sample_output=f"{args.tune_prefix}_{idx}_{spec['label']}.txt",
                sample_step=4,
                sample_ansi=False,
                no_display=True,
                log_file=f"{args.tune_prefix}_{idx}_{spec['label']}.json",
                seek=args.seek,
            )
            play_video_braille(**params)
            # Read the log back for metrics
            try:
                with open(params["log_file"], "r", encoding="utf-8") as fh:
                    info = json.load(fh)
            except Exception:
                info = {}
            info.update({
                "label": spec["label"],
                "log": params["log_file"],
                "sample": params["sample_output"],
                "cmd": f"python video_braille.py \"{args.video}\" --seek {args.seek or 0} --mode {params['mode']} --width {params['width']} --fps-limit {params['fps_limit']} --aspect {params['aspect']} --adaptive {params['adaptive']} --block-size {params['block_size']} --adaptive-c {params['adaptive_c']} --clahe {params['clahe_clip']} --gamma {params['gamma']} {'--edge-boost' if params['edge_boost'] else ''} --blur-kernel {params['blur_kernel']} --canny-low {params['canny_low']} --canny-high {params['canny_high']}",
            })
            # Compute a light heuristic score
            nb = info.get("metric_nonblank_ratio", 0.0)
            dd = info.get("metric_dots_density", 0.0)
            tr = info.get("metric_transitions_per_char", 0.0)
            fps = info.get("achieved_fps", 0.0)
            target_fps = params['fps_limit'] or fps
            s_nb = max(0.0, 1.0 - abs(nb - 0.35) / 0.35)
            s_dd = max(0.0, 1.0 - abs(dd - 0.22) / 0.22)
            s_tr = max(0.0, 1.0 - abs(tr - 0.30) / 0.30)
            s_fps = min(1.0, fps / max(1e-6, 0.9 * target_fps))
            score = round((0.4 * s_nb + 0.3 * s_dd + 0.3 * s_tr) * s_fps, 4)
            info["score"] = score
            results.append(info)

        # Sort and write summary
        results_sorted = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        summary_path = f"{args.tune_prefix}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump({"results": results_sorted}, fh, ensure_ascii=False, indent=2)
        print(f"Auto-tune complete. Summary written to: {summary_path}")
        print("Top candidates:")
        for r in results_sorted[:3]:
            print(f" - {r.get('label')}: score={r.get('score')}, fps={r.get('achieved_fps')}, sample={r.get('sample')}")
        return
    play_video_braille(
        path=args.video,
        width=args.width,
        mode=args.mode,
        threshold=args.threshold,
        invert=args.invert,
        fps_limit=args.fps_limit,
        aspect=args.aspect,
        adaptive=args.adaptive,
        block_size=args.block_size,
        adaptive_c=args.adaptive_c,
        gamma=args.gamma,
        clahe_clip=args.clahe,
        clahe_grid=args.clahe_grid,
        blur_kernel=args.blur_kernel,
        edge_boost=args.edge_boost,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        sample_seconds=args.sample_seconds,
        sample_output=args.sample_output,
        sample_step=args.sample_step,
        sample_ansi=args.sample_ansi,
        no_display=args.no_display,
        log_file=args.log_file,
        seek=args.seek,
        audio=args.audio,
        audio_volume=args.audio_volume,
        audio_player=args.audio_player,
        audio_resync=args.audio_resync,
        sync_to_audio=args.sync_to_audio and not args.no_sync_to_audio,
        debug=args.debug,
        on_error=args.on_error,
        max_bad_reads=args.max_bad_reads,
        backend=args.backend,
        checkpoint_secs=args.checkpoint_secs,
        restart_on_stall=args.restart_on_stall,
        max_restarts=args.max_restarts,
        restart_seek_back=args.restart_seek_back,
        stall_skip_forward=args.stall_skip_forward,
    )


if __name__ == "__main__":
    main()
