import cv2
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import requests
from ultralytics import YOLO
import time
import subprocess
from collections import defaultdict, Counter
import math
import dotenv
import os
import numpy as np

dotenv.load_dotenv()

connection = os.getenv('CONNECTION')
cloud_API = os.getenv('CLOUD_API')
if not connection:
    raise ValueError("CONNECTION environment is not set")

def list_cameras():
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE, text=True)
    output = result.stdout.split('\n')
    cameras = {}
    current_device = None

    for line in output:
        if line.endswith(':'):
            current_device = line[:-1]
        elif '/dev/video' in line:
            device_path = line.strip()
            if current_device and current_device not in cameras.values():
                cameras[device_path] = current_device

    return cameras


def select_camera():
    cameras = list_cameras()
    if not cameras:
        messagebox.showerror("Error", "No cameras found")
        return None
    camera_list = [f"{i}. {device} ({name})" for i, (device, name) in enumerate(cameras.items())]
    camera_selection = simpledialog.askinteger("Select Camera", f"Available cameras:\n" + "\n".join(camera_list) + "\nEnter camera number:")
    if camera_selection is not None and 0 <= camera_selection < len(cameras):
        selected_device = list(cameras.keys())[camera_selection]
        print(f"Selected camera: {selected_device}")
        return selected_device
    messagebox.showerror("Error", "Invalid camera selection")
    return None


def calculate_resistance(colors, color_values):
    if len(colors) < 4:
        return None

    first_digit = color_values[colors[0]]
    second_digit = color_values[colors[1]]
    multiplier = 10 ** color_values[colors[2]]
    resistance = (first_digit * 10 + second_digit) * multiplier

    return resistance

def mask_by_color_ranges(hsv_roi, color_ranges):
    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for hsv_ranges in color_ranges.values():
        for lower, upper in hsv_ranges:
            mask |= cv2.inRange(hsv_roi, lower, upper)
    return mask

class EduVision:
    def __init__(self, main):
        self.root = main
        self.root.title("EduVision")
        self.root.geometry("600x400")

        response = requests.post(f"{connection}/initialize")
        if response.status_code == 200:
            self.connection_id = response.json().get("connection_id")
            print(f"Connection ID: {self.connection_id}")
        else:
            raise Exception("Failed to initialize connection with the API")

        self.label = tk.Label(root, text=f"Connection: {self.connection_id}")
        self.label.pack(pady=5)

        self.model = YOLO("./validation/best.pt")
        self.running = False
        self.resistors_data = {}
        self.tree_items = {}
        self.object_ids = {}
        self.frozen_objects = {}
        self.results = []
        self.color_detections = defaultdict(list)
        self.resistor_frame_count = defaultdict(int)
        self.resistor_last_result = {}
        self.resistor_last_colors = {}
        self.last_seen = {}

        self.color_values = {
            "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
            "green": 5, "blue": 6, "violet": 7, "gray": 8, "white": 9, "gold": 10, "silver": 11
        }

        self.color_ranges = {
            "black": [((0, 0, 0), (180, 255, 60))],
            "brown": [((5, 80, 40), (20, 255, 200))],
            "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
            "orange": [((10, 100, 100), (25, 255, 255))],
            "yellow": [((20, 100, 100), (35, 255, 255))],
            "green": [((36, 60, 60), (85, 255, 255))],
            "blue": [((86, 60, 60), (130, 255, 255))],
            "violet": [((131, 60, 60), (160, 255, 255))],
            "gray": [((0, 0, 50), (180, 50, 180))],
            "white": [((0, 0, 200), (180, 40, 255))],
            "gold": [((15, 40, 120), (45, 180, 255))],
            "silver": [((0, 0, 140), (180, 40, 220))],
        }

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=5)

        self.global_resistance_label = tk.Label(root, text="Global Resistance: 0 Ohms")
        self.global_resistance_label.pack(pady=5)

        self.table = ttk.Treeview(root, columns=("ID", "Colors", "Resistance", "Orientation", "Action"),
                                  show="headings")
        self.table.heading("ID", text="ID")
        self.table.heading("Colors", text="Colors")
        self.table.heading("Resistance", text="Resistance")
        self.table.heading("Orientation", text="Orientation")
        self.table.heading("Action", text="Action")
        self.table.pack(pady=5, fill=tk.BOTH, expand=True)

        self.total_resistance = 0

        self.update_thread = threading.Thread(target=self.update_server)
        self.update_thread.start()

    def update_server(self):
        while True:
            if self.running:
                try:
                    resistors_list = [
                        {
                            "resistance": resistor_data["resistance"]
                        }
                        for yolo_id, resistor_data in self.resistors_data.items()
                    ]
                    requests.post(f"{connection}/update_resistors/{self.connection_id}", json=resistors_list)
                    requests.post(f"{connection}/update_resistance/{self.connection_id}",
                                  json={"new_resistance": self.total_resistance})
                except Exception as e:
                    print(f"Failed to update server: {e}")
            time.sleep(2)

    def start_detection(self):
        if not self.running:
            self.camera_index = select_camera()
            if self.camera_index is not None:
                self.running = True
                self.thread = threading.Thread(target=self.detect_objects)
                self.thread.start()
                self.cleanup_thread = threading.Thread(target=self.cleanup_old_entries)
                self.cleanup_thread.start()
        else:
            messagebox.showinfo("Info", "Detection is already running")

    def stop_detection(self):
        if self.running:
            self.running = False
            self.thread.join()
            self.cleanup_thread.join()
            messagebox.showinfo("Info", "Detection stopped")
        else:
            messagebox.showinfo("Info", "Detection is not running")

    def detect_objects(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            self.frame = frame

            success, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not success:
                print("Failed to encode frame as .jpg")
                continue

            try:
                response = requests.post(
                    f"{cloud_API}/detect_objects",
                    files={"file": ("frame.jpg", encoded_image.tobytes(), "image/jpeg")},
                    timeout=5
                )
                response.raise_for_status()
                self.results = response.json().get("detections", [])
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with cloud computing.")
                continue
            except ValueError as e:
                print(f"Error parsing JSON response.")
                continue

            if self.results:
                for detection in self.results:
                    yolo_id = detection.get("id")
                    if yolo_id is not None:
                        if yolo_id not in self.object_ids:
                            self.object_ids[yolo_id] = len(self.object_ids) + 1
                        self.last_seen[yolo_id] = time.time()

            self.recognize_resistor_colors(frame, self.results)
            cv2.imshow("EduVision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_resistor_colors(self, frame, results):
        expanded_boxes = []
        for obj in results:
            if obj.get("id") is not None:
                yolo_id = int(obj["id"])
                x1, y1, x2, y2 = map(int, obj["bbox"])
                ex1 = max(0, x1 - 2)
                ey1 = max(0, y1 - 2)
                ex2 = min(frame.shape[1] - 1, x2 + 2)
                ey2 = min(frame.shape[0] - 1, y2 + 2)
                expanded_boxes.append((ex1, ey1, ex2, ey2, yolo_id))

        for obj in results:
            if obj.get("id") is not None:
                yolo_id = int(obj["id"])
                x1, y1, x2, y2 = map(int, obj["bbox"])
                overlap = False
                for ex1, ey1, ex2, ey2, other_id in expanded_boxes:
                    if other_id != yolo_id:
                        if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
                            overlap = True
                            break
                if overlap:
                    continue

                if yolo_id in self.frozen_objects and self.frozen_objects[yolo_id]:
                    resistance = self.table.item(self.tree_items[yolo_id], "values")[2]
                    cv2.putText(frame, f"ID: {yolo_id}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, resistance, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    continue

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                color_mask = mask_by_color_ranges(hsv_roi, self.color_ranges)
                masked_gray = cv2.bitwise_and(gray, gray, mask=color_mask)

                thresh = cv2.adaptiveThreshold(masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                circles = cv2.HoughCircles(
                    thresh, cv2.HOUGH_GRADIENT, dp=1.5, minDist=5,
                    param1=50, param2=17, minRadius=4, maxRadius=10
                )

                recognized_colors = []
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        cx, cy, r = i
                        y1_patch = max(0, cy - r)
                        y2_patch = min(hsv.shape[0], cy + r)
                        x1_patch = max(0, cx - r)
                        x2_patch = min(hsv.shape[1], cx + r)
                        patch = hsv[y1_patch:y2_patch, x1_patch:x2_patch]

                        if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                            continue

                        color_count = {}
                        for color_name, hsv_ranges in self.color_ranges.items():
                            mask = None
                            for lower, upper in hsv_ranges:
                                current_mask = cv2.inRange(patch, lower, upper)
                                mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)
                            color_count[color_name] = cv2.countNonZero(mask)
                        best_color = max(color_count, key=color_count.get)
                        if color_count[best_color] > 0 and best_color not in recognized_colors:
                            recognized_colors.append(best_color)
                        cv2.circle(roi, (cx, cy), r, (0, 255, 0), 2)
                        cv2.putText(roi, best_color, (cx - r, cy - r), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                if len(recognized_colors) < 4:
                    for color_name, hsv_ranges in self.color_ranges.items():
                        mask = None
                        for lower, upper in hsv_ranges:
                            current_mask = cv2.inRange(hsv, lower, upper)
                            mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)
                        if cv2.countNonZero(mask) > 0 and color_name not in recognized_colors:
                            recognized_colors.append(color_name)

                self.color_detections[yolo_id].append(tuple(recognized_colors))
                most_common_colors = Counter(
                    [color for colors in self.color_detections[yolo_id] for color in colors]).most_common(4)
                best_colors = [color for color, _ in most_common_colors]

                if len(best_colors) >= 4:
                    self.color_detections[yolo_id].append(tuple(best_colors[:4]))
                    if len(self.color_detections[yolo_id]) > 5:
                        self.color_detections[yolo_id].pop(0)
                    self.resistor_frame_count[yolo_id] += 1

                    if self.resistor_frame_count[yolo_id] == 10:
                        most_common = Counter(self.color_detections[yolo_id]).most_common(1)
                        if most_common:
                            stable_colors = list(most_common[0][0])
                            self.resistor_last_colors[yolo_id] = stable_colors
                            result = calculate_resistance(stable_colors, self.color_values)
                            if result is not None:
                                resistance = result
                                self.resistor_last_result[yolo_id] = resistance
                                self.update_resistor_entry(yolo_id, stable_colors, resistance, "Unknown",
                                                           (x1, y1, x2, y2))
                        self.resistor_frame_count[yolo_id] = 0

                if yolo_id in self.resistor_last_result:
                    resistance = self.resistor_last_result[yolo_id]
                    cv2.putText(frame, f"ID: {yolo_id}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{resistance} Ohms", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def update_resistor_entry(self, yolo_id, colors, resistance, orientation, bbox):
        self.resistors_data[yolo_id] = {
            "colors": colors,
            "resistance": resistance,
            "orientation": orientation,
            "bbox": bbox
        }
        res_str = f"{resistance} Ohms"

        if yolo_id not in self.tree_items:
            row_id = self.table.insert("", "end", values=(yolo_id, colors, res_str, orientation, "Freeze"))
            self.tree_items[yolo_id] = row_id
            self.table.set(row_id, column="Action", value="Freeze")
        else:
            self.table.item(self.tree_items[yolo_id], values=(yolo_id, colors, res_str, orientation, "Freeze"))

        self.calculate_global_resistance()

    def cleanup_old_entries(self):
        while self.running:
            current_time = time.time()
            to_delete = [yolo_id for yolo_id, last_seen in self.last_seen.items() if current_time - last_seen > 2]
            for yolo_id in to_delete:
                if yolo_id in self.tree_items:
                    self.table.delete(self.tree_items[yolo_id])
                    del self.tree_items[yolo_id]
                    del self.resistors_data[yolo_id]
                    del self.last_seen[yolo_id]
            time.sleep(1)

    def calculate_global_resistance(self):
        global total_resistance
        series_resistors = {}
        parallel_resistors = {}
        is_infinity = False

        if len(self.resistors_data) > 1:
            for yolo_id, data in self.resistors_data.items():
                if self.is_parallel(yolo_id):
                    parallel_resistors[yolo_id] = data
                    is_infinity = False
                elif self.is_serial(yolo_id, parallel_resistors):
                    series_resistors[yolo_id] = data
                    is_infinity = False
                else:
                    is_infinity = True

        if not is_infinity:
            total_series_resistance = sum(resistor["resistance"] for resistor in series_resistors.values())
            total_parallel_resistance = sum(
                1 / resistor["resistance"] for resistor in parallel_resistors.values()) if parallel_resistors else 0
            total_parallel_resistance = 1 / total_parallel_resistance if total_parallel_resistance else 0

            total_resistance = total_series_resistance + total_parallel_resistance
            self.total_resistance = total_resistance
        else:
            total_resistance = float("inf")
            self.total_resistance = float("inf")

        self.global_resistance_label.config(text=f"Global Resistance: {total_resistance} Ohms")

    def is_parallel(self, yolo_id):
        x1, y1, x2, y2 = self.resistors_data[yolo_id]["bbox"]
        center1_x, center1_y = (x1 + x2) / 2, (y1 + y2) / 2
        for other_id, other_data in self.resistors_data.items():
            if other_id != yolo_id:
                ox1, oy1, ox2, oy2 = other_data["bbox"]
                center2_x, center2_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                distance = math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
                distance_threshold = 100
                if distance < distance_threshold:
                    if abs(center1_y - center2_y) < 10 and abs(center1_x - center2_x) > 15:
                        return True
        return False

    def is_serial(self, yolo_id, parallel_resistors):
        x1, y1, x2, y2 = self.resistors_data[yolo_id]["bbox"]
        center1_x, center1_y = (x1 + x2) / 2, (y1 + y2) / 2
        for other_id, other_data in self.resistors_data.items():
            if other_id != yolo_id:
                ox1, oy1, ox2, oy2 = other_data["bbox"]
                center2_x, center2_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                distance = math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
                distance_threshold = 200

                if distance < distance_threshold:
                    if abs(center1_x - center2_x) < 10 and abs(center1_y - center2_y) > 50:
                        return True
                    elif len(parallel_resistors) > 0:
                        for p_id, data in parallel_resistors.items():
                            if yolo_id != p_id:
                                px1, py1, px2, py2 = self.resistors_data[p_id]["bbox"]
                                pcenter_x, pcenter_y = (px1 + px2) / 2, (py1 + py2) / 2
                                if abs(center1_x - pcenter_x) < 20 and abs(center1_y - pcenter_y) > 30:
                                    return True


if __name__ == "__main__":
    root = tk.Tk()
    app = EduVision(root)
    root.mainloop()
