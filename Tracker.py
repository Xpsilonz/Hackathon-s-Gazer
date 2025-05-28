import cv2
import mediapipe as mp
import numpy as np
import win32api
import win32con
import win32gui
import win32ui
from PIL import Image, ImageGrab
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pytesseract
import threading
import queue
import requests
from datetime import datetime
import os
import json

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

screen_w = win32api.GetSystemMetrics(0)
screen_h = win32api.GetSystemMetrics(1)

running = True
calibration_data = {}
current_click_point = None

model_x = None
model_y = None

facemesh_data = [] 
clicked_coordinates = [] 
model = None 
scaler = StandardScaler()

# OCR and text recognition variables
ocr_enabled = True
ocr_queue = queue.Queue()
ocr_region_visible = True
recognized_text_history = []
current_recognized_text = ""
last_ocr_time = 0
ocr_cooldown = 0.5  
ocr_region_width = 300  # Horizontal width for better text reading
ocr_region_height = 100  # Shorter height for horizontal rectangle
text_feedback_enabled = True
speech_enabled = False

class TextLogger:
    def __init__(self, log_file="eye_tracker_text_log.json"):
        self.log_file = log_file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_text(self, text, position, confidence=None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "text": text,
            "position": position,
            "confidence": confidence
        }
        
        # Read existing logs
        logs = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        
        # Save logs
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        # Optional: Send to remote server
        self.send_to_server(log_entry)
    
    def send_to_server(self, log_entry):
        try:
            response = requests.post(
                'http://yodin2327.dothome.co.kr/gazer/pc/store_text.php',
                json={
                    'user_id': 3,
                    'content': log_entry['text'],
                    'position': log_entry['position'],
                    'confidence': log_entry['confidence'],
                    'timestamp': log_entry['timestamp']
                },
                timeout=10
            )
            if response.status_code != 200:
                print(f"Failed to send text to server: {response.status_code}")
        except Exception as e:
            print(f"Error sending text to server: {e}")

text_logger = TextLogger()

class OverlayDot:
    def __init__(self):
        self.hInstance = win32api.GetModuleHandle(None)
        self.className = "EyeTrackerOverlay"

        wndClass = win32gui.WNDCLASS()
        wndClass.lpfnWndProc = self.wndProc
        wndClass.hInstance = self.hInstance
        wndClass.lpszClassName = self.className
        wndClass.hbrBackground = win32gui.GetStockObject(win32con.WHITE_BRUSH)
        try:
            win32gui.RegisterClass(wndClass)
        except Exception:
            pass

        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST |
            win32con.WS_EX_TRANSPARENT | win32con.WS_EX_NOACTIVATE,
            self.className, "Gazer", win32con.WS_POPUP,
            0, 0, screen_w, screen_h, None, None, self.hInstance, None,
        )
        win32gui.SetLayeredWindowAttributes(self.hwnd, 0x00ffffff, 255, win32con.LWA_COLORKEY)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

    def wndProc(self, hwnd, msg, wParam, lParam):
        if msg == win32con.WM_NCHITTEST:
            return win32con.HTTRANSPARENT
        elif msg == win32con.WM_CLOSE:
            win32gui.DestroyWindow(hwnd)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wParam, lParam)

    def draw_dot(self, x, y, color=(0, 255, 0), size=5):
        hdc = win32gui.GetDC(self.hwnd)
        pen = win32gui.CreatePen(win32con.PS_SOLID, 8, win32api.RGB(color[0], color[1], color[2]))
        brush = win32gui.CreateSolidBrush(win32api.RGB(color[0], color[1], color[2]))
        win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, brush)
        win32gui.Ellipse(hdc, x - size, y - size, x + size, y + size)
        win32gui.ReleaseDC(self.hwnd, hdc)

    def draw_ocr_region(self, x, y, width=300, height=100):
        """Draw a horizontal rectangle around the OCR region for better text reading"""
        hdc = win32gui.GetDC(self.hwnd)
        pen = win32gui.CreatePen(win32con.PS_SOLID, 2, win32api.RGB(255, 255, 0))
        win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
        win32gui.Rectangle(hdc, x - width//2, y - height//2, x + width//2, y + height//2)
        win32gui.ReleaseDC(self.hwnd, hdc)

    def clear(self):
        hdc = win32gui.GetDC(self.hwnd)
        brush = win32gui.CreateSolidBrush(win32api.RGB(255, 255, 255))
        win32gui.FillRect(hdc, (0, 0, screen_w, screen_h), brush)
        win32gui.ReleaseDC(self.hwnd, hdc)

def get_active_window_info():
    """Get information about the currently active window"""
    try:
        # Get the active window handle
        hwnd = win32gui.GetForegroundWindow()
        
        # Get window rectangle
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        width = x2 - x
        height = y2 - y
        
        # Get window title
        window_title = win32gui.GetWindowText(hwnd)
        
        return {
            'hwnd': hwnd,
            'rect': (x, y, width, height),
            'title': window_title
        }
    except Exception as e:
        print(f"Error getting active window info: {e}")
        return None

def capture_active_window_region(gaze_x, gaze_y, ocr_width, ocr_height):
    """Capture a region from the active window only"""
    try:
        window_info = get_active_window_info()
        if not window_info:
            return None
            
        win_x, win_y, win_width, win_height = window_info['rect']
        
        # Check if gaze point is within the active window
        if not (win_x <= gaze_x <= win_x + win_width and win_y <= gaze_y <= win_y + win_height):
            print(f"Gaze point outside active window: {window_info['title']}")
            return None
        
        # Convert screen coordinates to window-relative coordinates
        relative_x = gaze_x - win_x
        relative_y = gaze_y - win_y
        
        # Calculate OCR region bounds relative to window
        ocr_left = max(0, relative_x - ocr_width // 2)
        ocr_top = max(0, relative_y - ocr_height // 2)
        ocr_right = min(win_width, relative_x + ocr_width // 2)
        ocr_bottom = min(win_height, relative_y + ocr_height // 2)
        
        # Capture the window
        hwnd = window_info['hwnd']
        
        # Get window device context
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        
        # Create bitmap
        dataBitMap = win32ui.CreateBitmap()
        region_width = ocr_right - ocr_left
        region_height = ocr_bottom - ocr_top
        
        dataBitMap.CreateCompatibleBitmap(dcObj, region_width, region_height)
        cDC.SelectObject(dataBitMap)
        
        # Copy the specific region from window to bitmap
        cDC.BitBlt((0, 0), (region_width, region_height), dcObj, (ocr_left, ocr_top), win32con.SRCCOPY)
        
        # Convert to PIL Image
        bmpinfo = dataBitMap.GetInfo()
        bmpstr = dataBitMap.GetBitmapBits(True)
        
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )
        
        # Clean up
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        print(f"Captured OCR region from active window: {window_info['title']}")
        return np.array(img)
        
    except Exception as e:
        print(f"Error capturing active window region: {e}")
        return None

# Modified capture_screen_region function
def capture_screen_region(x, y, width, height):
    """Capture a specific region from the active window only"""
    return capture_active_window_region(x, y, width, height)

# Alternative: Get window under cursor position
def get_window_under_cursor(cursor_x, cursor_y):
    """Get the window handle under the cursor position"""
    try:
        point = win32gui.POINT()
        point.x = cursor_x
        point.y = cursor_y
        hwnd = win32gui.WindowFromPoint(point)
        
        # Get the top-level window
        while True:
            parent = win32gui.GetParent(hwnd)
            if parent == 0:
                break
            hwnd = parent
            
        return hwnd
    except Exception as e:
        print(f"Error getting window under cursor: {e}")
        return None

def capture_window_under_cursor_region(gaze_x, gaze_y, ocr_width, ocr_height):
    """Capture region from the specific window under the gaze point"""
    try:
        hwnd = get_window_under_cursor(gaze_x, gaze_y)
        if not hwnd:
            return None
            
        # Get window rectangle
        rect = win32gui.GetWindowRect(hwnd)
        win_x, win_y, win_x2, win_y2 = rect
        win_width = win_x2 - win_x
        win_height = win_y2 - win_y
        
        # Convert to relative coordinates
        relative_x = gaze_x - win_x
        relative_y = gaze_y - win_y
        
        # Calculate region bounds
        ocr_left = max(0, relative_x - ocr_width // 2)
        ocr_top = max(0, relative_y - ocr_height // 2)
        ocr_right = min(win_width, relative_x + ocr_width // 2)
        ocr_bottom = min(win_height, relative_y + ocr_height // 2)
        
        # Capture window region (same as above)
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        
        dataBitMap = win32ui.CreateBitmap()
        region_width = ocr_right - ocr_left
        region_height = ocr_bottom - ocr_top
        
        dataBitMap.CreateCompatibleBitmap(dcObj, region_width, region_height)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (region_width, region_height), dcObj, (ocr_left, ocr_top), win32con.SRCCOPY)
        
        bmpinfo = dataBitMap.GetInfo()
        bmpstr = dataBitMap.GetBitmapBits(True)
        
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )
        
        # Clean up
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        window_title = win32gui.GetWindowText(hwnd)
        print(f"Captured OCR region from window under cursor: {window_title}")
        return np.array(img)
        
    except Exception as e:
        print(f"Error capturing window under cursor region: {e}")
        return None
    
def send_text_to_database(content):
    try:
        response = requests.post(
            'http://yodin2327.dothome.co.kr/gazer/pc/store_text.php',
            json={
                'user_id': 2,
                'content': content,
            },
            timeout=10
        )
        if response.status_code != 200:
            print(f"Failed to send text to database: {response.status_code}")
    except Exception as e:
        print(f"Error sending text to database: {e}")

def perform_ocr(image_region, position):
    """Perform OCR on the given image region"""
    global current_recognized_text
    
    try:
        if image_region is None:
            return ""
        
        # Convert to PIL Image if it's a numpy array
        if isinstance(image_region, np.ndarray):
            pil_image = Image.fromarray(image_region)
        else:
            pil_image = image_region
        
        # Enhance image for better OCR
        pil_image = pil_image.convert('RGB')
        
        # Perform OCR with confidence scores
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Extract text with confidence filtering
        texts = []
        confidences = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if text and conf > 30:  # Filter low confidence text
                texts.append(text)
                confidences.append(conf)
        
        recognized_text = ' '.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        if recognized_text:
            current_recognized_text = recognized_text
            
            # Log the recognized text
            text_logger.log_text(recognized_text, position, avg_confidence)
            
            # Add to history
            recognized_text_history.append({
                'text': recognized_text,
                'position': position,
                'timestamp': datetime.now(),
                'confidence': avg_confidence
            })
            
            # Keep only last 50 entries
            if len(recognized_text_history) > 50:
                recognized_text_history.pop(0)
            
            print(f"OCR Text at {position}: '{recognized_text}' (Confidence: {avg_confidence:.1f}%)")
            
            return recognized_text
        
    except Exception as e:
        print(f"OCR Error: {e}")
    
    return ""

def ocr_worker():
    """Background worker for OCR processing"""
    while running:
        try:
            if not ocr_queue.empty():
                x, y = ocr_queue.get(timeout=1)
                
                # Capture horizontal rectangular region around cursor for better text reading
                region = capture_screen_region(
                    x - ocr_region_width//2, 
                    y - ocr_region_height//2, 
                    ocr_region_width, 
                    ocr_region_height
                )
                
                if region is not None:
                    perform_ocr(region, (x, y))
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"OCR Worker Error: {e}")
        
        time.sleep(0.1)

def on_mouse(event, x, y, flags, param):
    global current_click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        screen_x = int(x * screen_w / param["frame_width"])
        screen_y = int(y * screen_h / param["frame_height"])
        current_click_point = (screen_x, screen_y)
        print(f"Clicked calibration point at {current_click_point}")

def train_calibration_model():
    global model_x, model_y, calibration_data

    all_features = []
    xs, ys = [], []

    for pt, feats_list in calibration_data.items():
        for f in feats_list:
            all_features.append(f)
            xs.append(pt[0])
            ys.append(pt[1])

    if len(all_features) < 10:
        print("Not enough data to train calibration model.")
        return False

    X = np.array(all_features)
    yx = np.array(xs)
    yy = np.array(ys)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    try:
        model_x = LinearRegression().fit(X_poly, yx)
        model_y = LinearRegression().fit(X_poly, yy)
        print("Calibration model trained with polynomial regression on accumulated data.")
        return True
    except Exception as e:
        print(f"Failed to train calibration model: {e}")
        return False

def build_deep_learning_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_position(features):
    global model_x, model_y

    if model_x is None or model_y is None:
        return screen_w // 2, screen_h // 2

    f_aug = np.array(features).reshape(1, -1)
    poly = PolynomialFeatures(degree=2)
    f_poly = poly.fit_transform(f_aug)
    try:
        x_pred = int(np.clip(model_x.predict(f_poly)[0], 0, screen_w))
        y_pred = int(np.clip(model_y.predict(f_poly)[0], 0, screen_h))
        return x_pred, y_pred
    except Exception:
        return screen_w // 2, screen_h // 2

def reset_calibration_data():
    global calibration_data, facemesh_data, clicked_coordinates, model, model_x, model_y
    calibration_data.clear()
    facemesh_data.clear()
    clicked_coordinates.clear()
    model = None
    model_x = None
    model_y = None
    print("Calibration data and model have been reset.")

def show_text_history():
    """Display recent recognized text"""
    print("\n=== Recent Recognized Text ===")
    for i, entry in enumerate(recognized_text_history[-10:], 1):
        print(f"{i}. [{entry['timestamp'].strftime('%H:%M:%S')}] "
              f"Pos({entry['position'][0]},{entry['position'][1]}): "
              f"'{entry['text'][:50]}...' (Conf: {entry['confidence']:.1f}%)")
    print("=" * 40)

def draw_rounded_rectangle(img, top_left, bottom_right, color, radius=10, thickness=2, fill=False):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()
    if fill:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_menu_button(frame):
    button_w, button_h = 120, 50
    x, y = 10, frame.shape[0] - button_h - 10
    draw_rounded_rectangle(frame, (x, y), (x + button_w, y + button_h), (30, 30, 30), radius=10, fill=True)
    cv2.putText(frame, " Menu", (x + 15, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def draw_menu_main(frame):
    overlay_w, overlay_h = 500, 300
    x = (frame.shape[1] - overlay_w) // 2
    y = (frame.shape[0] - overlay_h) // 2
    draw_rounded_rectangle(frame, (x, y), (x + overlay_w, y + overlay_h), (30, 30, 30), radius=12, fill=True)

    texts = [
        "Menu Options:",
        "1. Quit (press q)",
        "2. Reset Calibration (press r)",
        "3. Instructions (press i)",
        "4. Toggle OCR (press o)",
        "5. Show Text History (press x)",
        "6. OCR Settings (press s)",
        "Press ESC to close menu"
    ]
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (x + 20, y + 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_ocr_settings(frame):
    overlay_w, overlay_h = 580, 320
    x = (frame.shape[1] - overlay_w) // 2
    y = (frame.shape[0] - overlay_h) // 2
    draw_rounded_rectangle(frame, (x, y), (x + overlay_w, y + overlay_h), (30, 30, 30), radius=12, fill=True)

    texts = [
        "OCR Settings:",
        f"OCR Enabled: {'ON' if ocr_enabled else 'OFF'}",
        f"Region Width: {ocr_region_width}px",
        f"Region Height: {ocr_region_height}px",
        f"Cooldown: {ocr_cooldown}s",
        f"Text Feedback: {'ON' if text_feedback_enabled else 'OFF'}",
        f"OCR Region Frame: {'Visible' if ocr_region_visible else 'Hidden'}",
        "",
        "Controls:",
        "'t' - Toggle OCR | 'w' / Shift + 'w' - Adjust width",
        "'h' - Hide/Show OCR Frame | 'e' / Shift + 'e' - Adjust height",
        "'x' - Show Text History | 'b' - Back"
    ]
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (x + 20, y + 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def draw_instructions(frame):
    overlay_w, overlay_h = 600, 350
    x = (frame.shape[1] - overlay_w) // 2
    y = (frame.shape[0] - overlay_h) // 2
    draw_rounded_rectangle(frame, (x, y), (x + overlay_w, y + overlay_h), (30, 30, 30), radius=12, fill=True)

    texts = [
        "Instructions:",
        "EYE TRACKING:",
        "1. Click anywhere on cam feed to calibrate.",
        "2. Keep steady gaze on the dot for 2 sec.",
        "3. Calibration retrains after each point.",
        "",
        "TEXT RECOGNITION:",
        "Green dot shows where you're looking",
        "Yellow horizontal box shows OCR scan region",
        "Text is automatically recognized and logged",
        "Check console for recognized text output",
        "",
        "CONTROLS:",
        "'o' - Toggle OCR | 'x' - Show text history",
        "'f' - Fullscreen | ESC - Menu | 'h' - Hide/Show Frame",
        "'b' - Back "
    ]
    for i, text in enumerate(texts):
        font_size = 0.5 if text.startswith('•') or text.startswith("'") else 0.55
        cv2.putText(frame, text, (x + 20, y + 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)

def main():
    global running, current_click_point, calibration_data, facemesh_data, clicked_coordinates
    global model, scaler, ocr_enabled, last_ocr_time, ocr_region_width, ocr_region_height, text_feedback_enabled, ocr_region_visible

    # Start OCR worker thread
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Camera Feed", on_mouse, {"frame_width": frame_width, "frame_height": frame_height})

    cv2.imshow("Camera Feed", np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(1)

    hwnd = win32gui.FindWindow(None, "Camera Feed")
    if hwnd:
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    else:
        print("Warning: couldn't find window handle to maximize.")

    overlay = OverlayDot()

    LEFT_IRIS = [474, 475, 476, 477]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
                249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_IRIS = [469, 470, 471, 472]
    RIGHT_EYE = [33, 160, 158, 157, 154, 153, 144,
                385, 386, 387, 388, 389, 390, 391, 392, 393]

    smoothing = []
    MAX_SMOOTH = 8
    last_pos = (screen_w // 2, screen_h // 2)
    clicked_pos = None
    click_time = 0
    hold_duration = 2
    training_done_for_click = False

    menu_open = False
    menu_state = "main"
    fullscreen = False

    print("=== Enhanced Eye Tracker with Horizontal OCR Region ===")
    print("CALIBRATION:")
    print(" - Click anywhere in 'Camera Feed' window to calibrate.")
    print(f" - Keep steady gaze on the dot for {hold_duration} seconds.")
    print(" - Calibration model will be trained automatically after each point.")
    print()
    print("TEXT RECOGNITION:")
    print(" - Green dot shows where you're looking")
    print(" - Horizontal yellow rectangle optimized for text reading")
    print(" - Text around your gaze is automatically recognized")
    print(" - Recognized text is logged and displayed in console")
    print(" - Press 'h' to show/hide OCR region frame")
    print()
    print("CONTROLS:")
    print(" - 'c' - manually train calibration model")
    print(" - 'o' - toggle OCR on/off")
    print(" - 'x' - show text history")
    print(" - 'f' - toggle fullscreen")
    print(" - 'h' - hide/show OCR region frame")
    print(" - ESC - toggle menu")

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Retrying...")
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            features = None
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                if len(face.landmark) > max(LEFT_IRIS + LEFT_EYE):
                    iris_pts_left = [face.landmark[i] for i in LEFT_IRIS]
                    eye_pts_left = [face.landmark[i] for i in LEFT_EYE]
                    iris_pts_right = [face.landmark[i] for i in RIGHT_IRIS]
                    eye_pts_right = [face.landmark[i] for i in RIGHT_EYE]

                    if all(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in iris_pts_left + eye_pts_left + iris_pts_right + eye_pts_right):
                        eye_cx_left = np.mean([p.x for p in eye_pts_left])
                        eye_cy_left = np.mean([p.y for p in eye_pts_left])
                        iris_cx_left = np.mean([p.x for p in iris_pts_left])
                        iris_cy_left = np.mean([p.y for p in iris_pts_left])

                        eye_cx_right = np.mean([p.x for p in eye_pts_right])
                        eye_cy_right = np.mean([p.y for p in eye_pts_right])
                        iris_cx_right = np.mean([p.x for p in iris_pts_right])
                        iris_cy_right = np.mean([p.y for p in iris_pts_right])

                        features_left = [iris_cx_left - eye_cx_left, iris_cy_left - eye_cy_left]
                        features_right = [iris_cx_right - eye_cx_right, iris_cy_right - eye_cy_right]
                        features = features_left + features_right

            if current_click_point:
                clicked_pos = current_click_point
                click_time = time.time()
                current_click_point = None
                training_done_for_click = False

            now = time.time()
            if clicked_pos and (now - click_time) <= hold_duration:
                if features is not None:
                    calibration_data.setdefault(clicked_pos, []).append(features)
                    facemesh_data.append(features)
                    clicked_coordinates.append(clicked_pos)

                overlay.clear()
                overlay.draw_dot(*clicked_pos, color=(255, 0, 0))  # Red during calibration
                cv2.circle(frame, (int(clicked_pos[0] * frame_width / screen_w), int(clicked_pos[1] * frame_height / screen_h)),
                           15, (0, 0, 255), 4)
                cv2.putText(frame, f"Collecting data... {int(hold_duration - (now - click_time))}s",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif clicked_pos and not training_done_for_click:
                trained = train_calibration_model()
                training_done_for_click = True
                overlay.clear()
                overlay.draw_dot(*clicked_pos, color=(0, 255, 0))
                cv2.circle(frame, (int(clicked_pos[0] * frame_width / screen_w), int(clicked_pos[1] * frame_height / screen_h)),
                           15, (0, 255, 0), 4)
                cv2.putText(frame, "Calibration model trained at clicked point.",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if len(clicked_coordinates) % 10 == 0 and len(clicked_coordinates) > 9:
                    model = build_deep_learning_model(len(facemesh_data[0]))
                    X = np.array(facemesh_data)
                    y = np.array(clicked_coordinates)
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y, epochs=10, batch_size=10)
                    model.save('eyetrackermodel.h5')
                    print("Deep learning model trained and saved.")

            else:
                if features:
                    pred_x, pred_y = predict_position(features)
                else:
                    pred_x, pred_y = last_pos

                smoothing.append((pred_x, pred_y))
                if len(smoothing) > MAX_SMOOTH:
                    smoothing.pop(0)

                smooth_x = int(np.median([p[0] for p in smoothing]))
                smooth_y = int(np.median([p[1] for p in smoothing]))

                overlay.clear()
                
                overlay.draw_dot(smooth_x, smooth_y, color=(0, 255, 0))
                
                if ocr_enabled and ocr_region_visible:
                    overlay.draw_ocr_region(smooth_x, smooth_y, ocr_region_width, ocr_region_height)
                    
                if ocr_enabled and now - last_ocr_time >= ocr_cooldown:
                    ocr_queue.put((smooth_x, smooth_y))
                    last_ocr_time = now

                last_pos = (smooth_x, smooth_y)

            cv2.putText(frame, "Click to calibrate points. Look around for text recognition.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Clicks: {len(calibration_data)} | Press: ESC for menu",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if current_recognized_text and text_feedback_enabled:
                text_display = current_recognized_text[:60] + "..." if len(current_recognized_text) > 60 else current_recognized_text
                cv2.putText(frame, f"Text: {text_display}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            draw_menu_button(frame)

            if menu_open:
                if menu_state == "main":
                    draw_menu_main(frame)
                elif menu_state == "ocr_settings":
                    draw_ocr_settings(frame)
                elif menu_state == "instructions":
                    draw_instructions(frame)

            cv2.imshow("Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if not menu_open:
                if key == ord('c'):
                    train_calibration_model()
                elif key == ord('o'):
                    ocr_enabled = not ocr_enabled
                    print(f"OCR {'enabled' if ocr_enabled else 'disabled'}")
                elif key == ord('h'):
                    ocr_region_visible = not ocr_region_visible
                    print(f"OCR Region Frame {'visible' if ocr_region_visible else 'hidden'}")
                elif key == 27:
                    menu_open = True
                    menu_state = "main"
                elif key == ord('q'):
                    running = False
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                if menu_state == "main":
                    if key == ord('q'):
                        running = False
                    elif key == ord('r'):
                        reset_calibration_data()
                        menu_state = "main"
                    elif key == ord('i'):
                        menu_state = "instructions"
                    elif key == ord('s'):
                        menu_state = "ocr_settings"
                    elif key == 27:
                        menu_open = False
                    elif key == ord('o'):
                        ocr_enabled = not ocr_enabled
                    elif key == ord('f'):
                        fullscreen = not fullscreen
                        if fullscreen:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    elif key == ord('f'):
                        fullscreen = not fullscreen
                        if fullscreen:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif menu_state == "ocr_settings":
                    if key == ord('b') or key == 27:
                        menu_state = "main"
                    elif key == ord('o'):
                        ocr_enabled = not ocr_enabled
                    elif key == ord('W'):
                        ocr_region_width += 10
                        print(f"OCR Region Width: {ocr_region_width}px")
                    elif key == ord('w'):
                        ocr_region_width = max(10, ocr_region_width - 10)
                        print(f"OCR Region Width: {ocr_region_width}px")
                    elif key == ord('E'):
                        ocr_region_height += 10
                        print(f"OCR Region Height: {ocr_region_height}px")
                    elif key == ord('e'):
                        ocr_region_height = max(10, ocr_region_height - 10)
                        print(f"OCR Region Height: {ocr_region_height}px")
                    elif key == ord('h'):
                        ocr_region_visible = not ocr_region_visible
                        print(f"OCR Region Frame {'visible' if ocr_region_visible else 'hidden'}")
                    elif key == ord('q'):
                        running = False
                    elif key == ord('f'):
                        fullscreen = not fullscreen
                        if fullscreen:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif menu_state == "instructions":
                    if key == ord('b') or key == 27: 
                        menu_state = "main"
                    elif key == ord('f'):
                        fullscreen = not fullscreen
                        if fullscreen:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()