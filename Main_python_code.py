from ultralytics import YOLO
import cv2
import time
import pandas as pd
import serial
import threading
import numpy as np
from collections import defaultdict
import os

# Create defects directory if it doesn't exist
if not os.path.exists('defects'):
    os.makedirs('defects')

# ========================
# CONFIGURATION
# ========================
CONFIG = {
    'model': 'YOLO_11m.pt',
    'confidence_threshold': 0.85,
    'cam_index': 1,
    'manual_focus_value': 208,
    'detection_zone_width': 60,
    'defective_labels': ['Defective_Cap', 'Flipped_Cap'],
    
    'max_conveyor_speed': 400,
    'distance_to_ejector_mm': 180,
    'conveyor_speed_to_mms_factor': 0.5,
    'ejection_cooldown_s': 5.0,
    
    'servo_open_degree': 70,
    'servo_time_to_open_s': 0.3,
    'servo_time_to_close_s': 1.0,
    'servo_hold_duration_s': 3.5,
    
    'serial_port': 'COM3',
    'baud_rate': 115200,
    'log_to_csv': True,
    'csv_filename': 'detection_log.csv',
    
    'color_map': {
        'system_on':(0,255,0),
        'system_off':(0,0,255),
        'Defective_Cap': (0, 0, 255),
        'Flipped_Cap': (0, 165, 255),
        'OK_Cap': (0, 255, 0),
        'Outside_Zone': (128, 128, 128)
    },
    
    'idle_timeout_s': 30,  # System turns off if no detections for 30 seconds
}

# ========================
# SYSTEM STATE
# ========================
STATE = {
    'system_on': False,
    'conveyor_speed': 0,
    'next_object_id': 1,
    'counted_track_ids': set(),
    'pending_ejections': [],
    'last_ejection_time': 0,
    'ser': None,
    'esp32_status_message': 'Initializing...',
    'defect_counter': defaultdict(int),
    'last_detection_time': None  # Track time of last detection
}

# ========================
# INITIALIZATION & COMMUNICATION
# ========================
# Initializes the YOLO model, camera, and serial connection to the ESP32, handling errors gracefully.
# Returns the model and camera objects if successful, or None if initialization fails.
def initialize_system():
    print('System initializing...')
    try:
        print(f"Loading {CONFIG['model']}...")
        model = YOLO(CONFIG['model'])
        cap = cv2.VideoCapture(CONFIG['cam_index'])
        if not cap.isOpened(): raise RuntimeError('Could not open video source')
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, CONFIG['manual_focus_value'])
        print('Model loaded successfully')
    except Exception as e:
        print(f'FATAL: Camera or Model error: {e}'); return None, None
    try:
        print(f'Establishing Serial communication via {CONFIG['serial_port']}...')
        STATE['ser'] = serial.Serial(CONFIG['serial_port'], CONFIG['baud_rate'], timeout=1)
        time.sleep(2)
        print('Communication with ESP32 established successfully')
    except Exception as e:
        print(f'WARNING: Serial connection error: {e}'); STATE['ser'] = None
    print('--- System Initialized Successfully ---')
    return model, cap

def send_command(cmd):
    if STATE['ser']:
        try:
            STATE['ser'].write(f"{cmd}\n".encode())
        except serial.SerialException as e:
            print(f'FATAL: Serial connection lost! {e}')
            STATE['ser'] = None

def send_initial_config():
    print('Sending initial configuration to ESP32...')
    send_command(f"A{CONFIG['servo_open_degree']}")
    send_command(f"O{int(CONFIG['servo_time_to_open_s'] * 1000)}")
    send_command(f"C{int(CONFIG['servo_time_to_close_s'] * 1000)}")
    send_command(f"H{int(CONFIG['servo_hold_duration_s'] * 1000)}")

# Runs in a separate thread to listen for messages from the ESP32 over serial.
# Updates the ESP32 status message for the UI and handles connection errors.
def serial_handler():
    while STATE['ser'] is not None:
        try:
            if STATE['ser'].in_waiting > 0:
                line = STATE['ser'].readline().decode('utf-8').strip()
                if line:
                    print(f'ESP32: {line}')
                    if line.startswith('STATUS:'):
                        STATE['esp32_status_message'] = line.replace('STATUS: ', '')
        except (serial.SerialException, OSError):
            print('Serial connection lost. Listener thread stopping.')
            break
        time.sleep(0.05)

# ========================
# CORE LOGIC
# ========================
# Processes each video frame using YOLO to detect and track caps.
# Draws detection zone lines, logs detections, saves defect images, and schedules ejections for defective caps.
def process_frame(frame, model, log_data):
    if not STATE['system_on']: return frame, log_data
    h, w = frame.shape[:2]
    zx = w // 2  # Center of the frame
    ll = zx - CONFIG['detection_zone_width']  # Left boundary of detection zone
    rl = zx + CONFIG['detection_zone_width']  # Right boundary of detection zone
    cv2.line(frame, (ll, 0), (ll, h), (0, 0, 255), 2)  # Draw left zone line
    cv2.line(frame, (rl, 0), (rl, h), (0, 0, 255), 2)  # Draw right zone line
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False, conf=CONFIG['confidence_threshold'])  # Run YOLO detection with tracking

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            tid = int(box.id[0])  # Track ID for persistent object tracking
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            lbl = model.names[int(box.cls[0])]  # Label of the detected object
            inside = x1 > ll and x2 < rl  # Check if box is within detection zone

            if inside and tid not in STATE['counted_track_ids']:
                STATE['counted_track_ids'].add(tid)  # Mark object as counted
                mapped = 'Damaged_Cap' if lbl == 'Defective_Cap' else lbl  # Map label for UI and CSV consistency
                log_data.append((time.strftime('%H:%M:%S'), STATE['next_object_id'], mapped))  # Log detection
                STATE['defect_counter'][lbl] += 1
                STATE['next_object_id'] += 1

                if lbl in CONFIG['defective_labels']:
                    crop = frame[y1:y2, x1:x2]  # Crop defect area for image saving
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    fn = f"defects/defect_{ts}_{mapped}_{STATE['next_object_id']-1}.jpg"  # Use current ID for filename
                    cv2.imwrite(fn, crop)
                    print(f"Saved defect image: {fn}")

                if lbl in CONFIG['defective_labels'] and (time.time() - STATE['last_ejection_time']) > CONFIG['ejection_cooldown_s']:
                    sp = STATE['conveyor_speed'] * CONFIG['conveyor_speed_to_mms_factor']  # Calculate speed in mm/s
                    if sp > 0:
                        d = CONFIG['distance_to_ejector_mm'] / sp  # Calculate delay to reach ejector
                        et = time.time() + d  # Schedule ejection time
                        STATE['pending_ejections'].append(et)
                        print(f"Defect detected! Scheduling ejector in {d:.2f}s.")
                        STATE['esp32_status_message'] = f"Ejecting in {d:.2f}s..."

            col = CONFIG['color_map'].get(lbl) if inside else CONFIG['color_map']['Outside_Zone']
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{'Damaged_Cap' if lbl == 'Defective_Cap' else lbl} {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Check for idle timeout
    if STATE['system_on'] and STATE['last_detection_time'] is not None:
        if time.time() - STATE['last_detection_time'] > CONFIG['idle_timeout_s']:
            print("Idle timeout reached. Shutting down.")
            STATE['system_on'] = False
            STATE['conveyor_speed'] = 0
            send_command('X')
            STATE['esp32_status_message'] = 'System idle, stopped.'

    return frame, log_data

# ========================
# UI & INPUT
# ========================
# Renders the user interface panel below the video feed, displaying system status, counts, and controls.
# Organizes information in a vertical layout for clarity.
def draw_ui(frame):
    h, w = frame.shape[:2]
    panel = np.zeros((220, w, 3), dtype=np.uint8)
    txt_color = (255, 255, 255)

    status = 'ON' if STATE['system_on'] else 'OFF'

    cv2.putText(panel, f"System: {status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG['color_map']['system_on'] if STATE['system_on'] else CONFIG['color_map']['system_off'] , 2) 
    cv2.putText(panel, f"Conveyor speed: {STATE['conveyor_speed']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)
    cv2.putText(panel, f"Ejector: {STATE['esp32_status_message']}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)

    ok = STATE['defect_counter'].get('OK_Cap', 0)
    fl = STATE['defect_counter'].get('Flipped_Cap', 0)
    dmg = STATE['defect_counter'].get('Defective_Cap', 0)
    total_def = fl + dmg
    total_caps = sum(STATE['defect_counter'].values())

    cv2.putText(panel, f"OK: {ok}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG['color_map']['OK_Cap'], 2)
    cv2.putText(panel, f"Flipped: {fl}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG['color_map']['Flipped_Cap'], 2)
    cv2.putText(panel, f"Damaged: {dmg}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG['color_map']['Defective_Cap'], 2)
    cv2.putText(panel, f"Total Defects: {total_def}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)
    cv2.putText(panel, f"Total Caps: {total_caps}", (200, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)

    ctrl = "Controls: [s]tart/stop  [+/-] conveyor speed  [t]rigger  [r]eset  [q]quit"
    cv2.putText(panel, ctrl, (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 1)
    return np.vstack([frame, panel])

# Handles keyboard inputs to control the system (start/stop, speed, ejector, reset, quit).
# Updates system state and sends commands to the ESP32 as needed.
def handle_key_input(key):
    if key == ord('q'):  # Quit the program
        return False
    if key == ord('s'):  # Toggle system on/off
        STATE['system_on'] = not STATE['system_on']
        if STATE['system_on']:
            STATE['conveyor_speed'] = 200
            send_command(f"V{STATE['conveyor_speed']}")  # Start conveyor
            STATE['esp32_status_message'] = 'System started, ejector ready.'
        else:
            STATE['conveyor_speed'] = 0
            send_command('X')  # Stop conveyor
            STATE['esp32_status_message'] = 'System stopped.'
    if key == ord('r'):  # Reset detection counters
        STATE['defect_counter'].clear()
        STATE['counted_track_ids'].clear()
        STATE['next_object_id'] = 1
        print('Counters reset')
    if STATE['system_on']:
        if key in [ord('+'), ord('=')]:  # Increase conveyor speed
            STATE['conveyor_speed'] = min(CONFIG['max_conveyor_speed'], STATE['conveyor_speed'] + 50)
            send_command(f"V{STATE['conveyor_speed']}")
        elif key == ord('-'):  # Decrease conveyor speed
            STATE['conveyor_speed'] = max(0, STATE['conveyor_speed'] - 50)
            send_command(f"V{STATE['conveyor_speed']}")
        elif key == ord('t'):  # Manually trigger ejector
            send_command('T')
    return True

# ========================
# MAIN PROGRAM
# ========================
if __name__ == '__main__':
    model, cap = initialize_system()
    if model and cap:
        send_initial_config()
        if STATE['ser']:
            threading.Thread(target=serial_handler, daemon=True).start()
        log_data = []
        print("--- System ready. Press 's' to start, 'q' to quit. ---")

        while True:
            # Exit if camera or serial connection is lost
            if not cap.isOpened() or STATE['ser'] is None:
                print('Camera or Serial disconnected. Shutting down.')
                break
            ret, frame = cap.read()
            if not ret:
                print('Error: Failed to grab frame.')
                continue
            processed, log_data = process_frame(frame.copy(), model, log_data)
            now = time.time()
            # Trigger any scheduled ejections
            if STATE['pending_ejections'] and now >= STATE['pending_ejections'][0]:
                print(">>> Sending 'T' command to ESP32!")
                send_command('T')
                STATE['last_ejection_time'] = now
                STATE['pending_ejections'].pop(0)
            disp = draw_ui(processed)
            cv2.imshow('Cap Detection System', disp)
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not handle_key_input(key):
                break

    print('Shutting down...')
    send_command('X')
    if STATE['ser']: STATE['ser'].close()
    if 'cap' in locals() and cap.isOpened(): cap.release()
    cv2.destroyAllWindows()
    if CONFIG['log_to_csv'] and 'log_data' in locals() and log_data:
        pd.DataFrame(log_data, columns=['Time', 'ID', 'Label']).to_csv(CONFIG['csv_filename'], index=False)
        print(f"Log saved to {CONFIG['csv_filename']}")
    print('Shutdown complete.')