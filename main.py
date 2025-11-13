import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from pid_controller import PID  # Assumes pid_controller.py is in the same directory
import pyvjoy
import keyboard
from ultralytics import YOLO
import math
import itertools
import sys
import os

# Conclude setting / general reprocessing / plots / metrices / datasets
# Make sure 'utils' is a folder in your project with utils.py
try:
    from utils.utils import \
        time_synchronized, select_device, increment_path, \
        LoadImages, lane_line_mask
except ImportError:
    print("Error: Failed to import from 'utils.utils'.")
    print("Please ensure 'utils/utils.py' exists and is in your PYTHONPATH.")
    print("You can download it from a YOLOPv2 repository if missing.")
    sys.exit(1)


# --- WINDOW SETTINGS ---
MONITOR_X_OFFSET = 0
MONITOR_Y_OFFSET = 0
WINDOW_NAME = 'Lane-only (Left & Right)'
# -----------------------
# -------------------------
# Configuration / settings
# -------------------------

# --- YOLOv12 (Object) Config ---
# [FIX 1] Corrected typo from yolo12n.pt
MODEL_PATH = "yolo12n.pt"
CONF_THRESH = 0.3
IMGSZ = 640

# --- Screen Capture Settings ---
MONITOR_NUMBER = 1
RES_W, RES_H = 1280, 720
RES = (RES_W, RES_H)
CAPTURE_REGION = (0, 0, RES_W, RES_H)

# --- Analysis ROI ---
ROI_TOP_FRAC = 0.25
ROI_BOTTOM_FRAC = 0.75
# Fallback width for "danger zone" (as % of screen width)
DANGER_ZONE_WIDTH_FRAC = 0.4

# --- Tracking / growth thresholds ---
MATCH_DISTANCE_RATIO = 0.06
STALENESS_TIME = 1.0
GROWTH_SMOOTH_ALPHA = 0.4
AREA_STOP_THRESHOLD = 0.80

# --- Danger scoring constants ---
SCALE = 1e5
HARD_THRESH = 300.0
BRAKE_THRESH = 150.0
SLOW_THRESH = 30.0
MIN_THRESH = 5.0

# --- Throttle/Brake Settings (TUNE THESE) ---
CRUISE_THROTTLE = 10000
SLOW_THROTTLE = 5000
FULL_BRAKE = 32768
FULL_THROTTLE = 32768

# --- vJoy Constants ---
VJOY_NEUTRAL = 16384
VJOY_MAX = 32768
VJOY_MIN = 0

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='screen', help='source')  # file/folder, 0 for webcam, "screen"
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/detect', help='(unused here, kept for consistency)')
    parser.add_argument('--name', default='exp_screen', help='(unused here, kept for consistency)')
    parser.add_argument('--exist-ok', action='store_true', help='(unused here, kept for consistency)')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')

    return parser

def reset_controller(j):
    j.data.wAxisX = 16384  # left stick X
    j.data.wAxisY = 16384  # neutral Y
    j.data.wAxisZ = 0      # left trigger (Brake)
    j.data.wAxisZRot = 0   # right trigger (Gas)
    j.reset()
    j.update()
    return j

def update_controller(j, lx, lt = 0, rt = 0):
        j.data.wAxisX = lx
        j.data.wAxisY = 16384   # neutral Y
        j.data.wAxisZ = lt      # left trigger (Brake)
        j.data.wAxisZRot = rt   # right trigger (Gas)
        j.update() # Send new values to vJoy
        return j


# -------------------------
# Utilities
# -------------------------
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) * (y2 - y1))

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

# -------------------------
# Track Class
# -------------------------
class Track:
    _id_iter = itertools.count(1)
    
    def __init__(self, bbox, label, area, t, centroid):
        self.id = next(Track._id_iter)
        self.bbox = bbox
        self.label = label
        self.last_area = area
        self.last_time = t
        self.centroid = centroid
        self.growth_rate = 0.0
        self.smoothed_growth = 0.0
        self.priority = 0.0
        self.last_seen = t

    def update(self, bbox, area, t, centroid, screen_area):
        dt = max(1e-6, t - self.last_time)
        raw_growth = ((area - self.last_area) / screen_area) / dt
        self.smoothed_growth = GROWTH_SMOOTH_ALPHA * raw_growth + (1 - GROWTH_SMOOTH_ALPHA) * self.smoothed_growth
        self.growth_rate = raw_growth
        self.bbox = bbox
        self.last_area = area
        self.last_time = t
        self.centroid = centroid
        self.last_seen = t

    def mark_seen(self, t):
        self.last_seen = t


def error_calc(left_lane, right_lane, target_lane, w, h):
    """
    Positive => steer right; Negative => steer left.
    Uses bottom-half rows with quadratic weights toward the very bottom.
    Adapts virtual lane width from observed lane-gap median.
    """
    # --- choose rows: lists are bottom->top, so index 0 is bottom ---
    top_rows = h//4 + h//2
    bottom_rows = h//4
    L = left_lane[bottom_rows: top_rows]
    R = right_lane[bottom_rows: top_rows]
    C = target_lane[bottom_rows: top_rows]
    H = top_rows - bottom_rows  # length actually used

    # --- compute adaptive virtual lane width from observed gaps ---
    gaps = []
    for i in range(H):
        li, ri = L[i], R[i]
        if li and ri:
            lx, rx = li[0], ri[0]
            if rx > lx:
                gaps.append(rx - lx)
    if gaps:
        gap_med = float(np.median(gaps))
        # Virtual half-width ~ 40–60% of median gap, clamped
        pix_threshold = int(np.clip(0.5 * gap_med, 60, 0.15 * w))
    else:
        # fallback if we couldn't measure the gap
        pix_threshold = max(int(0.06 * w), 50)

    # --- quadratic weights emphasizing bottom rows ---
    weights = (np.linspace(1.0, 0.0, H) ** 2 + 1e-6).astype(float)

    err_sum = 0.0
    w_sum = 0.0

    for i in range(H):
        left_point  = L[i]
        right_point = R[i]
        cx = C[i][0]
        row_w = weights[i]

        if left_point and right_point:
            lx, rx = left_point[0], right_point[0]
            if rx <= lx:
                continue  # bad geometry; skip

            err_sum += (rx + lx - 2*cx) * row_w
            w_sum += row_w

        elif left_point and not right_point:
            lx = left_point[0]
            rx = lx + pix_threshold

            err_sum += (rx + lx - 2*cx) * row_w
            w_sum += row_w

        elif right_point and not left_point:
            rx = right_point[0]
            lx = rx - pix_threshold

            err_sum += (rx + lx - 2*cx) * row_w
            w_sum += row_w
        # else: no data -> no contribution

    if w_sum == 0.0:
        return 0

    err_norm = err_sum / w_sum
    return int(np.clip(err_norm, -4 * pix_threshold, 4 * pix_threshold))


def draw_left_right_points_only(ll_seg_mask, out_shape):
    """
    No preprocessing, no interpolation, no smoothing.
    Just find per-row innermost left and right lane pixels and draw *points*
    on a black canvas.
    """
    h, w, _ = out_shape
    canvas = np.zeros(out_shape, dtype=np.uint8)
    # Lists to store points
    left_lane = []
    right_lane = []
    # Ensure boolean/binary
    mask = (ll_seg_mask > 0)

    center_x = w // 2

    # Scan every row (bottom->top or top->bottom; doesn't matter for points)
    for y in range(h - 1, -1, -1):
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            left_lane.append(())
            right_lane.append(())
            continue

        # Left = innermost pixel to the left of center
        left_side = xs[xs < center_x]
        if left_side.size > 0:
            lx = int(left_side.max())
            # draw a small dot for left lane (blue)
            left_lane.append((lx, y))
            cv2.circle(canvas, (lx, y), radius=1, color=(255, 0, 0), thickness=-1)  # BGR
        else:
            left_lane.append(())

        # Right = innermost pixel to the right of center
        right_side = xs[xs >= center_x]
        if right_side.size > 0:
            rx = int(right_side.min())
            # draw a small dot for right lane (green)
            right_lane.append((rx, y))
            cv2.circle(canvas, (rx, y), radius=1, color=(0, 255, 0), thickness=-1)  # BGR
        else:
            right_lane.append(())

    return canvas, left_lane, right_lane


def detect():
    # parse options
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    # device + model
    device = select_device(opt.device)
    try:
        model = torch.jit.load(weights[0], map_location=device).to(device).eval()
    except Exception as e:
        print(f"Error loading YOLOPv2 model from {weights[0]}: {e}")
        print("Make sure the --weights argument points to your YOLOPv2 model.")
        return
        
    print("YOLOPv2 (Lane) Model loaded")
    half = device.type != 'cpu'
    if half:
        model.half()

    print("Loading YOLOv12 (Object) model...")
    try:
        object_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLOv12 model from {MODEL_PATH}: {e}")
        print("Make sure the MODEL_PATH constant is correct.")
        return
        
    print("YOLOv8 model loaded")
    # data loader
    dataset = LoadImages(source, img_size=imgsz, stride=32)

    # ------------------ Video recording setup ------------------
    save_video = not getattr(opt, "nosave", False)
    writer_raw = None
    writer_overlay = None
    if save_video:
        fps = getattr(dataset, "fps", None) or getattr(dataset, "frame_rate", None) or 30.0
        try:
            fps = float(fps)
        except:
            fps = 10.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ts = time.strftime("%Y%m%d_%H%M%S")

        out_dir = Path(increment_path(Path("runs/recordings") / "exp", exist_ok=True))
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = str(out_dir / f"record_raw_{ts}.mp4")
        overlay_path = str(out_dir / f"record_overlay_{ts}.mp4")

        writer_raw = {"path": raw_path, "fourcc": fourcc, "fps": fps, "writer": None}
        writer_overlay = {"path": overlay_path, "fourcc": fourcc, "fps": fps, "writer": None}
        print(f"[REC] Will save raw -> {raw_path}")
        print(f"[REC] Will save overlay -> {overlay_path}")


    # State variables
    tracks = {}
    paused = False
    last_ts = time.perf_counter()
    prev_time = 0.0
    last_pause_toggle = 0.0
    PAUSE_DEBOUNCE_TIME = 0.3
    # window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, MONITOR_X_OFFSET, MONITOR_Y_OFFSET)
    cv2.resizeWindow(WINDOW_NAME, RES_W, RES_H)

    # warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    center_lane = []

    # --- Initialize PID controller ---
    pid = PID(Kp=30.0, Ki=0.001, Kd=30.0)
    pid.reset()
    print(" PID Controller ready.")

    # --- Initialize virtual controller ---
    print("Initializing virtual controller...")
    try:
        j = pyvjoy.VJoyDevice(1)
        reset_controller(j)
        print("Virtual Controller ready.")
    except Exception as e:
        print("Controller initialization failed:", e)
        print("Make sure vJoy is installed and Device 1 is enabled.")
        if save_video:
            print("[REC] Recording disabled due to controller init failure.")
        return

    # Timing for dt
    last_ts = time.perf_counter()
    smoothed_error = 0.0
    SMOOTHING_FACTOR = 0.75

    # --- ADDED: State for pause functionality ---
    paused = False
    t_key_was_pressed = False
    print("\n--- Controls ---")
    print(" T: Pause / Resume")
    print(" S: Brake (Hold)")
    print(" Q: Quit (in OpenCV window)")
    print("----------------\n")
    # ------------------------------------------

    try:
        # main loop
        for path, img, im0s, vid_cap in dataset:

            # --- ADDED: Pause/Resume Logic ---
            if keyboard.is_pressed('t'):
                if not t_key_was_pressed:  # On new key press
                    paused = not paused
                    t_key_was_pressed = True
                    if paused:
                        print("\n[PAUSED] - Resetting controller. Press 'T' to resume.")
                        reset_controller(j)  # CRITICAL: Stop the car
                        pid.reset()
                    else:
                        print("\n[RESUMING]")
            else:
                t_key_was_pressed = False  # Key is released, ready for next press

            # If paused, skip the rest of the loop but keep window responsive
            if paused:
                # We must keep checking for 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting from paused state.")
                    break  # Break main loop
                time.sleep(0.01)
                continue
            # --- END PAUSE LOGIC ---

            frame_start = time.perf_counter()
            # to tensor
            img_t = torch.from_numpy(img).to(device)
            img_t = img_t.half() if half else img_t.float()
            img_t /= 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            # inference
            t1 = time_synchronized()
            # model returns: [pred, anchor_grid], seg, ll  (we only need ll)
            _, _, ll = model(img_t)
            t2 = time_synchronized()

            # lane mask (H, W) ints
            ll_seg_mask = lane_line_mask(ll)

            # --- [FIX 2] STEERING LOGIC (WITH MASK RESIZE) ---
            h, w = im0s.shape[:2]
            
            # Resize the low-resolution lane mask (e.g., 640x384) to the
            # high-resolution screen capture size (e.g., 1280x720)
            ll_seg_mask_resized = cv2.resize(ll_seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Now, process points using the *resized* mask
            canvas, left_lane, right_lane = draw_left_right_points_only(ll_seg_mask_resized, (h, w, 3))

            center_x = w // 2
            center_lane = [(center_x, y) for y in range(h - 1, -1, -1)]

            steer_error = error_calc(left_lane, right_lane, center_lane, w, h)

            # Apply EMA smoothing
            smoothed_error = (SMOOTHING_FACTOR * smoothed_error) + ((1.0 - SMOOTHING_FACTOR) * steer_error)

            # Compute dt using full frame time
            now = time.perf_counter()
            dt = max(1e-3, now - last_ts)  # guard tiny/zero
            last_ts = now

            # Feed the SMOOTH error to the PID
            steer_axis = pid.control(smoothed_error, dt)
            steering_axis = int(np.clip(steer_axis, 0, 32768))  # Center around neutral

            # --- END STEERING LOGIC ---

            # make a black canvas same size as original im0s and draw only lane dots
            screen_area = h * w
            max_dim = max(w, h)
            match_dist_thresh = max_dim * MATCH_DISTANCE_RATIO
            static_center_x = w // 2

            # Define ROI (for object detection)
            roi_top_y = int(h * ROI_TOP_FRAC)
            roi_bottom_y = int(h * ROI_BOTTOM_FRAC)

            # Danger zone
            danger_zone_half_w = (w * DANGER_ZONE_WIDTH_FRAC) / 2.0
            danger_zone_left = static_center_x - danger_zone_half_w
            danger_zone_right = static_center_x + danger_zone_half_w

            results = object_model.predict(im0s, conf=CONF_THRESH, imgsz=IMGSZ, verbose=False)

            # --- COLLECT DETECTIONS ---
            detections = []
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
                    label = object_model.names[cls].lower()
                    detections.append({
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "conf": float(conf),
                        "label": label
                    })

            # --- TRACK MATCHING ---
            curr_time = time.perf_counter()

            det_centroids = [box_center(det["bbox"]) for det in detections]
            used_track_ids = set()
            assigned = []

            for di, det in enumerate(detections):
                cx, cy = det_centroids[di]
                best_tid, best_dist = None, float("inf")
                for tid, tr in tracks.items():
                    if tid in used_track_ids:
                        continue
                    tx, ty = tr.centroid
                    dist = math.hypot(cx - tx, cy - ty)
                    if dist < best_dist:
                        best_dist, best_tid = dist, tid
                if best_dist <= match_dist_thresh and best_tid is not None:
                    assigned.append((best_tid, di))
                    used_track_ids.add(best_tid)

            # Update assigned tracks
            for tid, di in assigned:
                det = detections[di]
                tracks[tid].update(bbox=det["bbox"], area=box_area(det["bbox"]),
                                   t=curr_time, centroid=box_center(det["bbox"]),
                                   screen_area=screen_area)
                detections[di]["_assigned"] = True

            # Create new tracks
            for det in detections:
                if not det.get("_assigned", False):
                    tr = Track(bbox=det["bbox"], label=det["label"],
                               area=box_area(det["bbox"]), t=curr_time,
                               centroid=box_center(det["bbox"]))
                    tracks[tr.id] = tr

            # Remove stale tracks
            stale_ids = [tid for tid, tr in tracks.items() if (curr_time - tr.last_seen) > STALENESS_TIME]
            for tid in stale_ids:
                del tracks[tid]

            # --- THREAT EVALUATION ---
            stop_flag = False
            commands = []

            for tid, tr in tracks.items():
                x1, y1, x2, y2 = tr.bbox
                area = tr.last_area
                area_frac = area / screen_area
                cx, cy = tr.centroid

                # Check if object is in vertical ROI
                if not (roi_top_y < cy < roi_bottom_y):
                    continue

                # Check danger zone (center)
                if not (danger_zone_left < cx < danger_zone_right):
                    continue

                # Draw bounding box (on im0s, the original image)
                color = (0, 255, 0)
                if tr.smoothed_growth > 0.002:
                    color = (0, 0, 255)
                elif tr.smoothed_growth > 0.0005:
                    color = (0, 200, 255)

                cv2.rectangle(im0s, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                info = f"ID:{tr.id} {tr.label} gr:{tr.smoothed_growth:.6f}"
                cv2.putText(im0s, info, (int(x1), max(12, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Immediate stop
                if area_frac >= AREA_STOP_THRESHOLD:
                    stop_flag = True
                    commands.append((float('inf'), f"BRAKE/STOP - ID {tr.id}", tr.id, VJOY_NEUTRAL))
                    continue

                # Threat priority
                growth_val = max(0.0, tr.smoothed_growth)
                cumulation = (growth_val ** 2) * (area_frac * 100.0) * SCALE
                tr.priority = cumulation

                if cumulation < MIN_THRESH:
                    continue

                # HYBRID STEERING LOGIC (Modified)
                if cx < static_center_x:
                    direction = "RIGHT"
                    avoid_steer = 20480  # 25% right
                elif cx > static_center_x:
                    direction = "LEFT"
                    avoid_steer = 12288  # 25% left
                else:
                    direction = "CENTER"
                    avoid_steer = VJOY_NEUTRAL

                if cumulation >= HARD_THRESH:
                    action = f"HARD {direction} (ID {tr.id})"
                    if direction == "RIGHT":
                        avoid_steer = 24576
                    elif direction == "LEFT":
                        avoid_steer = 8192
                elif cumulation >= BRAKE_THRESH:
                    action = f"BRAKE (ID {tr.id})"
                    avoid_steer = VJOY_NEUTRAL
                else:
                    action = f"SLOW {direction} (ID {tr.id})"

                commands.append((cumulation, action, tr.id, avoid_steer))

            # --- FINAL COMMAND RESOLUTION ---
            final_command = "ACTION: NONE"
            color = (50, 200, 50)
            final_steer = steering_axis

            if stop_flag:
                stop_cmds = [c for c in commands if c[0] == float('inf')]
                if stop_cmds:
                    final_command = f"ACTION: {stop_cmds[0][1]}"
                    final_steer = stop_cmds[0][3]
                    color = (0, 0, 255)
            elif commands:
                commands.sort(key=lambda x: x[0], reverse=True)
                highest_priority_cmd = commands[0]
                final_command = f"ACTION: {highest_priority_cmd[1]}"
                final_steer = highest_priority_cmd[3]
                color = (0, 0, 255)

            # --- THROTTLE/BRAKE LOGIC ---
            if "BRAKE" in final_command or "STOP" in final_command or "HARD" in final_command:
                throttle_output = 0
                brake_output = FULL_BRAKE
                pid.reset()
            elif "SLOW" in final_command:
                throttle_output = SLOW_THROTTLE
                brake_output = 0
                pid.reset()
            elif "ACTION: NONE" in final_command:
                throttle_output = CRUISE_THROTTLE
                brake_output = 0
            else:  # "HARD" avoidance
                throttle_output = CRUISE_THROTTLE
                brake_output = 0

            # Manual override
            if keyboard.is_pressed('w'):
                throttle_output = FULL_THROTTLE
            if keyboard.is_pressed('s'):
                brake_output = FULL_BRAKE
                throttle_output = 0

            # Debug text
            debug_text = f'Err: {smoothed_error} | Str: {final_steer:.0f} | Thr: {throttle_output:.0f} | Brk: {brake_output:.0f}'

            update_controller(j, lx=final_steer, lt=brake_output, rt=throttle_output)

            # Draw center line on the black canvas
            cv2.line(canvas, (center_x, 0), (center_x, h), (255, 255, 255), 1)
            print(f'Done. ({t2 - t1:.3f}s)')

            # Draw debug text onto canvas
            cv2.putText(canvas, debug_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, final_command, (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

            # Blend im0s and canvas
            display_img = cv2.addWeighted(im0s, 0.8, canvas, 1.0, 0)

            # ------ initialize VideoWriters on first frame if saving ------
            if save_video:
                if writer_raw["writer"] is None:
                    h_f, w_f = im0s.shape[:2]
                    # cv2 expects (width, height)
                    writer_raw["writer"] = cv2.VideoWriter(writer_raw["path"], writer_raw["fourcc"], writer_raw["fps"], (w_f, h_f))
                if writer_overlay["writer"] is None:
                    h_f2, w_f2 = display_img.shape[:2]
                    writer_overlay["writer"] = cv2.VideoWriter(writer_overlay["path"], writer_overlay["fourcc"], writer_overlay["fps"], (w_f2, h_f2))
                # write frames (ensure frames are BGR uint8)
                try:
                    writer_raw["writer"].write(im0s)
                    writer_overlay["writer"].write(display_img)
                except Exception as e:
                    print("[REC] Write failed:", e)
            # --------------------------------------------------------------

            # show the *blended* image
            cv2.imshow(WINDOW_NAME, display_img)
            
            # --- [FIX 3] CLEANED UP END-OF-LOOP LOGIC ---
            # 'T' (pause) is handled at the top of the loop.
            # We just check for 'q' in the waitKey to quit.
            if keyboard.is_pressed('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting loop.")
    except Exception as e:
        print("Exception in detect loop:", e)
        import traceback
        traceback.print_exc()
    finally:
        # --- ADDED: Cleanup (release video writers) ---
        if save_video:
            for w in (writer_raw, writer_overlay):
                try:
                    if isinstance(w, dict) and w.get("writer") is not None:
                        w["writer"].release()
                        print(f"[REC] Saved: {w['path']}")
                except Exception as e:
                    print("[REC] Error releasing writer:", e)
        # Reset controller already present
        print("Exiting... Resetting controller.")
        try:
            reset_controller(j)
        except Exception as e:
            print("Error resetting controller on exit:", e)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)
    
    # Check for pid_controller.py
    if not os.path.exists('pid_controller.py'):
        print("Error: 'pid_controller.py' not found.")
        print("Please make sure 'pid_controller.py' is in the same directory.")
    else:
        with torch.no_grad():
            detect()