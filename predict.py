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

"""
Loads environment variables from a '.env' file and retrieves specific variables.

Results:
    - Loads environment variables into the program's environment.
    - Retrieves the 'CONNECTION' and 'CLOUD_API' variables.
    - Raises a 'ValueError' if the 'CONNECTION' variable is not set.

Behaviour:
    - Uses the 'dotenv' library to load variables from a '.env' file.
    - Ensures that the 'CONNECTION' variable is mandatory for the program to run.
"""

dotenv.load_dotenv()

connection = os.getenv('CONNECTION')
cloud_API = os.getenv('CLOUD_API')
if not connection:
    raise ValueError("CONNECTION environment is not set")

def list_cameras():
    """
    Lists all available cameras connected to the system.

    Results:
        - Returns a dictionary with video device paths as keys and device names as values.
        - If no cameras are found, the dictionary will be empty.

    Behaviour:
        - List of connected video devices.
        - Parses the command output to extract device names and their corresponding paths.
        - Ensures that each device name is unique in the returned dictionary.
    """
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
    """
    Prompts the user to select a camera from the available list.

    Results:
        - Returns the selected camera device path as a string.
        - If no cameras are found, displays an error message and returns None.
        - If the user makes an invalid selection, displays an error message and returns None.

    Behaviour:
        - Calls 'list_cameras' to retrieve a dictionary of available cameras.
        - Displays a dialog box for the user to select a camera by entering its number.
        - Validates the user's selection and returns the corresponding device path.
    """
    cameras = list_cameras()
    if not cameras:
        messagebox.showerror("Error", "No cameras found")
        return None
    camera_list = [f"{i}. {device} ({name})" for i, (device, name) in enumerate(cameras.items())]
    camera_selection = simpledialog.askinteger("Select Camera", f"Available cameras:\n" + "\n".join(camera_list) + "\nEnter camera number:")
    if camera_selection is not None and 0 <= camera_selection < len(cameras):
        selected_device = list(cameras.keys())[camera_selection]
        return selected_device
    messagebox.showerror("Error", "Invalid camera selection")
    return None

def calculate_resistance(colors, color_values):
    """
    Calculates the resistance value of a resistor based on its color bands.

    Args:
        colors: A list of color names representing the bands on the resistor.
        color_values: A dictionary mapping color names to their corresponding numeric values.

    Results:
        - Returns the calculated resistance value as an integer.
        - If the input list of colors has fewer than 4 elements, returns None.

    Behaviour:
        - Extracts the first two digits from the color bands.
        - Determines the multiplier based on the third color band.
        - Computes the resistance value
    """
    if len(colors) < 4:
        return None

    first_digit = color_values.get(colors[0], 0)
    second_digit = color_values.get(colors[1], 0)
    multiplier = 10 ** color_values.get(colors[2], 0)
    resistance = (first_digit * 10 + second_digit) * multiplier

    return resistance

def mask_by_color_ranges(hsv_roi, color_ranges):
    """
    Masks the input HSV region of interest based on specified color ranges.

    Args:
        hsv_roi: HSV region of interest.
        color_ranges: A dictionary where keys are color names and values are lists of HSV range tuples.

    Results:
        - Returns a binary mask where pixels within the specified color ranges are set to 255, and others are set to 0.

    Behaviour:
        - Iterates through the provided color ranges.
        - Applies the 'cv2.inRange' function for each HSV range to create a mask.
        - Combines all masks using a bitwise OR operation to produce the final mask.
    """
    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for hsv_ranges in color_ranges.values():
        for lower, upper in hsv_ranges:
            mask |= cv2.inRange(hsv_roi, lower, upper)
    return mask

def apply_white_balance(image):
    """
    Adjusts the white balance of the input image.

    Args:
        image: A BGR image represented as a NumPy array.

    Results:
        - Returns a new image with adjusted white balance.
        - Balances the blue and red channels relative to the green channel.

    Behaviour:
        - Splits the image into blue, green, and red channels.
        - Calculates the mean intensity of each channel.
        - Scales the blue and red channels to match the green channel's mean intensity.
        - Clips the scaled values to the valid range and merges the channels back.
    """
    b, g, r = cv2.split(image)
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    b_scale = g_mean / b_mean if b_mean != 0 else 1
    r_scale = g_mean / r_mean if r_mean != 0 else 1
    b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
    r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

class EduVision:
    def __init__(self, main):
        """
        Initializes application.

        Args:
            main: Tkinter root window.

        Results:
            - Initializes the EduVision application UI and internal data structures.
            - Establishes a connection with the server and retrieves a connection ID.
            - Loads the YOLO model for object detection.
            - Starts a background thread to update the server with resistor data.

        Behaviour:
            - Sets up the main application window with a title and dimensions.
            - Creates UI elements such as labels, buttons, and a table for displaying resistor data.
            - Initializes internal dictionaries and variables for tracking resistors and their properties.
            - Starts a background thread to continuously send updates to the server.
        """
        self.root = main
        self.root.title("EduVision")
        self.root.geometry("600x400")

        response = requests.post(f"{connection}/initialize")
        if response.status_code == 200:
            self.connection_id = response.json().get("connection_id")
        else:
            raise Exception("Failed to initialize connection with the API")

        self.label = tk.Label(self.root, text=f"Connection: {self.connection_id}")
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
            "green": 5, "blue": 6, "violet": 7, "gray": 8, "white": 9,
            "gold": -1, "silver": -2
        }

        self.color_ranges = {
            "black": [((0, 0, 0), (180, 255, 60))],
            "brown": [((5, 40, 20), (35, 255, 200))],
            "red": [((0, 60, 60), (15, 255, 255)), ((160, 60, 60), (180, 255, 255))],
            "orange": [((10, 60, 60), (30, 255, 255))],
            "yellow": [((25, 60, 60), (45, 255, 255))],
            "green": [((35, 30, 30), (85, 255, 255))],
            "blue": [((90, 30, 30), (135, 255, 255))],
            "violet": [((130, 30, 30), (165, 255, 255))],
            "gray": [((0, 0, 60), (180, 50, 180))],
            "white": [((0, 0, 190), (180, 40, 255))],
            "gold": [((20, 40, 60), (45, 255, 220))],
            "silver": [((0, 0, 140), (180, 40, 230))],
        }

        self.start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=5)

        self.global_resistance_label = tk.Label(self.root, text="Global Resistance: 0 Ohms")
        self.global_resistance_label.pack(pady=5)

        self.table = ttk.Treeview(self.root, columns=("ID", "Colors", "Resistance", "Orientation", "Action"),
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
        """
        Sending data with the resistors and their resistances values for each instance to the server.

        Results:
            - Sends the list of resistors and their resistances to the server.
            - Updates the total resistance on the server.

        Behaviour:
            - Continuously runs in a loop while the application is active.
            - Checks if detection is running.
            - Posts the resistors data and total resistance to the server every 2 seconds.
            - Handles exceptions by printing an error message if the server update fails.
        """
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
        """
        Starts the detection process by selecting a camera and initializing detection.

        Results:
            - Initializes the detection process by selecting a camera.
            - Starts threads for object detection and cleanup of old entries.
            - Displays a message if detection is already running.

        Behaviour:
            - Checks if detection is already running.
            - Prompts the user to select a camera if not running.
            - Starts the detection and cleanup threads if a camera is selected.
            - Displays an informational message if detection is already active.
        """
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
        """
        Stops the detection process and cleans up resources.

        Results:
            - Stops the detection process if it is currently running.
            - Joins the threads responsible for detection and cleanup.
            - Displays a message indicating that detection has stopped.

        Behaviour:
            - Checks if the detection process is running.
            - If running, sets 'self.running' to False and joins the threads.
            - If not running, displays a message indicating that detection is not active.
        """
        if self.running:
            self.running = False
            self.thread.join()
            self.cleanup_thread.join()
            messagebox.showinfo("Info", "Detection stopped")
        else:
            messagebox.showinfo("Info", "Detection is not running")

    def detect_objects(self):
        """
        Detect resistors using YOLO model and process video frames from the camera.

        Results:
            - Captures video frames from the selected camera.
            - Tracks objects in the video frames using the YOLO model.
            - Calls 'recognize_resistor_colors' to process detected objects and extract resistor color information.

        Behaviour:
            - Continuously reads frames from the camera while 'self.running' is True.
            - Stops processing if the camera cannot be opened or if the user presses the "q" key.
            - Releases the camera and closes all OpenCV windows when stopped.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame = frame

            self.results = self.model.track(source=frame, conf=0.1, show_conf=False, persist=True, device="cpu")
            # With GPU
            # self.results = self.model.track(source=frame, conf=0.1, show_conf=False, persist=True, device="0")

            for result in self.results:
                for obj in result.boxes:
                    if obj.id is not None:
                        yolo_id = int(obj.id.item())
                        if yolo_id not in self.object_ids:
                            self.object_ids[yolo_id] = len(self.object_ids) + 1
                        self.last_seen[yolo_id] = time.time()

            self.recognize_resistor_colors(frame, self.results)
            cv2.imshow("EduVision", frame)
            cv2.waitKey()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_resistor_colors(self, frame, results):
        """
        Args:
            frame: The current video frame captured from the camera.
            results: The detection results from the YOLO model, containing bounding boxes and object IDs.

        Results:
            - Updates the 'self.color_detections' dictionary with detected colors for each resistor.
            - Updates the 'self.resistor_last_colors' and 'self.resistor_last_result' dictionaries with stable colors and calculated resistance values.
            - Calls 'update_resistor_entry' to update the UI and internal data structures.

        Behaviour:
            - Processes each detected resistor in the frame.
            - Expands bounding boxes slightly to account for detection inaccuracies.
            - Determines the orientation of each resistor.
            - Segments the resistor into bands and identifies the dominant color in each band.
            - Handles overlapping resistors by skipping them.
            - Calculates the resistance value based on the detected colors.
            - Ensures stable color detection by averaging results over multiple frames.
        """
        frame = apply_white_balance(frame)
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

        expanded_boxes = []
        for result in results:
            for obj in result.boxes:
                if obj.id is not None:
                    yolo_id = int(obj.id.item())
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    ex1 = max(0, x1 - 2)
                    ey1 = max(0, y1 - 2)
                    ex2 = min(frame.shape[1] - 1, x2 + 2)
                    ey2 = min(frame.shape[0] - 1, y2 + 2)
                    expanded_boxes.append((ex1, ey1, ex2, ey2, yolo_id))

        for result in results:
            for obj in result.boxes:
                if obj.id is not None:
                    yolo_id = int(obj.id.item())
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

                    width = x2 - x1
                    height = y2 - y1
                    orientation = "Horizontal" if width > height else "Vertical"

                    overlap = False


                    for ex1, ey1, ex2, ey2, other_id in expanded_boxes:
                        if other_id != yolo_id:
                            if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
                                overlap = True
                                break
                    if overlap:
                        continue

                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    largest_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]
                    if angle < -45:
                        angle += 90

                    center = (roi.shape[1] // 2, roi.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_roi = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]))
                    rotated_hsv = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2HSV)

                    recognized_colors = []
                    num_bands = 4
                    if orientation == "Horizontal":
                        segment_width = rotated_hsv.shape[1] // num_bands
                        for band in range(num_bands):
                            start_x = band * segment_width
                            end_x = (band + 1) * segment_width
                            band_roi = rotated_hsv[:, start_x:end_x]
                            if band_roi.size == 0:
                                continue
                            color_counts = Counter()
                            for i in range(band_roi.shape[0]):
                                for j in range(band_roi.shape[1]):
                                    pixel_hsv = band_roi[i, j]
                                    for color_name, hsv_ranges in self.color_ranges.items():
                                        for lower, upper in hsv_ranges:
                                            if (lower[0] <= pixel_hsv[0] <= upper[0] and
                                                lower[1] <= pixel_hsv[1] <= upper[1] and
                                                lower[2] <= pixel_hsv[2] <= upper[2]):
                                                color_counts[color_name] += 1
                                                break
                            if color_counts:
                                dominant_color = color_counts.most_common(1)[0][0]
                                recognized_colors.append(dominant_color)
                                band_x = x1 + start_x + segment_width // 2
                                band_y = y1 + (y2 - y1) // 2
                                if 0 <= band_x < frame.shape[1] and 0 <= band_y < frame.shape[0]:
                                    cv2.circle(frame, (int(band_x), int(band_y)), 3, (0, 0, 255), -1)
                    else:
                        segment_height = rotated_hsv.shape[0] // num_bands
                        for band in range(num_bands):
                            start_y = band * segment_height
                            end_y = (band + 1) * segment_height
                            band_roi = rotated_hsv[start_y:end_y, :]
                            if band_roi.size == 0:
                                continue
                            color_counts = Counter()
                            for i in range(band_roi.shape[0]):
                                for j in range(band_roi.shape[1]):
                                    pixel_hsv = band_roi[i, j]
                                    for color_name, hsv_ranges in self.color_ranges.items():
                                        for lower, upper in hsv_ranges:
                                            if (lower[0] <= pixel_hsv[0] <= upper[0] and
                                                lower[1] <= pixel_hsv[1] <= upper[1] and
                                                lower[2] <= pixel_hsv[2] <= upper[2]):
                                                color_counts[color_name] += 1
                                                break
                            if color_counts:
                                dominant_color = color_counts.most_common(1)[0][0]
                                recognized_colors.append(dominant_color)
                                band_x = x1 + (x2 - x1) // 2
                                band_y = y1 + start_y + segment_height // 2
                                if 0 <= band_x < frame.shape[1] and 0 <= band_y < frame.shape[0]:
                                    cv2.circle(frame, (int(band_x), int(band_y)), 3, (0, 0, 255), -1)

                    if recognized_colors:
                        if len(recognized_colors) >= 4:
                            if recognized_colors[0] in ["gold", "silver"]:
                                recognized_colors = recognized_colors[::-1]
                            elif recognized_colors[-1] not in ["gold", "silver"]:
                                pass

                    self.color_detections[yolo_id].append(tuple(recognized_colors))
                    most_common_colors = Counter(
                        [color for colors in self.color_detections[yolo_id] for color in colors]).most_common(4)
                    best_colors = [color for color, _ in most_common_colors]

                    if len(best_colors) >= 4:
                        self.color_detections[yolo_id].append(tuple(best_colors[:4]))
                        if len(self.color_detections[yolo_id]) > 5:
                            self.color_detections[yolo_id].pop(0)
                        self.resistor_frame_count[yolo_id] += 1

                        if self.resistor_frame_count[yolo_id] == 3:
                            most_common = Counter(self.color_detections[yolo_id]).most_common(1)
                            if most_common:
                                stable_colors = list(most_common[0][0])
                                self.resistor_last_colors[yolo_id] = stable_colors
                                result = calculate_resistance(stable_colors, self.color_values)
                                if result is not None:
                                    resistance = result
                                    self.resistor_last_result[yolo_id] = resistance
                                    self.update_resistor_entry(yolo_id, stable_colors, resistance, orientation,
                                                              (x1, y1, x2, y2))
                            self.resistor_frame_count[yolo_id] = 0


    def update_resistor_entry(self, yolo_id, colors, resistance, orientation, bbox):
        """
        Updates the resistor entry in the internal data structures and the UI table.

        Args:
            yolo_id: The unique identifier of the resistor.
            colors: A list of detected colors for the resistor.
            resistance: The calculated resistance value of the resistor.
            orientation: The orientation of the resistor.
            bbox: The bounding box of the resistor as a tuple.

        Results:
            - Updates the 'self.resistors_data' dictionary with the resistor's details.
            - Adds or updates the resistor entry in the UI table.
            - Calls 'calculate_global_resistance' to update the total resistance.

        Behaviour:
            - If the resistor is new, it creates a new entry in the UI table.
            - If the resistor already exists, it updates the existing entry.
            - Ensures the global resistance is recalculated after updating the entry.
        """
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
        """
        Removes old entries from the dictionaries, if they have not been updated within the last 2 seconds.

        Behaviour:
            - Continuously checks for resistors that have not been updated recently.
            - Deletes their corresponding entries from the UI table and internal data structures.
            - Runs in a loop while 'self.running' is True, with a 1-second delay between iterations.
        """
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
        """
        Calculates the total global resistance of the resistors in the system.

        Results:
            Updates the 'self.total_resistance' attribute with the calculated global resistance.
            Updates the 'self.global_resistance_label' with the calculated resistance value.

        Behaviour:
            - Iterates through the resistors in 'self.resistors_data'.
            - Classifies resistors as either in series or parallel using 'is_parallel' and 'is_serial' methods.
            - Calculates the total resistance for series and parallel configurations.
            - Handles cases where the resistance is infinite.
            - Updates the UI label to display the calculated global resistance.
        """
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
        """
        Determines if a resistor is in a parallel configuration with another resistor.

        Args:
            yolo_id: The unique identifier of the resistor being checked.

        Returns:
            bool: True if the resistor is in a parallel configuration, False otherwise.

        Behaviour:
            - Calculates the center of the bounding box for the given resistor.
            - Iterates through all other resistors to compare their positions.
            - Checks if the distance between the centers of the resistors is below a threshold.
            - Verifies if the resistors are aligned parallel.
        """
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
        """
        Determines if a resistor is in a serial configuration with another resistor.

        Args:
            yolo_id: The unique identifier of the resistor being checked.
            parallel_resistors: A dictionary of resistors identified as being in parallel configuration.

        Returns:
            bool: True if the resistor is in a serial configuration, False otherwise.

        Behaviour:
            - Calculates the center of the bounding box for the given resistor.
            - Iterates through all other resistors to compare their positions.
            - Checks if the distance between the centers of the resistors is below a threshold.
            - Verifies if the resistors are aligned serial.
            - Also checks if the resistor is aligned with any parallel resistors.
        """
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
        return False

if __name__ == "__main__":
    root = tk.Tk()
    app = EduVision(root)
    root.mainloop()