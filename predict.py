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
from sklearn.cluster import KMeans


dotenv.load_dotenv()
connection = os.getenv('CONNECTION')
if not connection:
    raise ValueError("CONNECTION environment is not set")

def list_cameras():
    """
    Lists all available cameras connected to the system.

    This function uses the `v4l2-ctl` command to retrieve a list of video devices
    and their associated names. It parses the output to create a dictionary
    mapping device paths (e.g., `/dev/video0`) to their corresponding device names.

    Returns:
        dict: A dictionary where keys are device paths and values are device names.
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
    Prompts the user to select a camera from the list of available cameras.

    This function retrieves a list of connected cameras using the `list_cameras` function,
    displays the available options in a dialog box, and allows the user to select a camera
    by entering its corresponding number.

    Returns:
        str: The device path of the selected camera (e.g., '/dev/video0') if a valid selection is made.
        None: If no cameras are found or the selection is invalid.
    """
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
    """
        Calculates the resistance value of a resistor based on its color bands.

        Args:
            colors (list): A list of color names representing the bands on the resistor.
            color_values (dict): A dictionary mapping color names to their corresponding numeric values.

        Returns:
            float: The calculated resistance value in ohms, or None if the input is invalid.
    """
    if len(colors) < 4:
        return None

    first_digit = color_values[colors[0]]
    second_digit = color_values[colors[1]]
    multiplier = 10 ** color_values[colors[2]]
    resistance = (first_digit * 10 + second_digit) * multiplier

    return resistance

class EduVision:
    def __init__(self, main):
        """
        Initializes the EduVision application.

        Args:
            main (tk.Tk): The main Tkinter root window.

        Behavior:
            - Sets up the main application window with a title, size, and UI components.
            - Initializes the connection to the API and retrieves a connection ID.
            - Loads the YOLO model for object detection.
            - Initializes data structures for resistor detection and tracking.
            - Defines color ranges for resistor bands and background detection.
            - Creates UI elements such as buttons, labels, and a table for displaying results.
            - Starts a background thread to update the server with resistor data.
        """
        self.root = main
        self.root.title("EduVision")
        self.root.geometry("600x400")
        self.camera_index = None
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

        # Initialize color detection data structures
        self.color_detections = {}
        self.resistor_frame_count = {}
        self.resistor_last_colors = {}
        self.resistor_last_result = {}

        # Color ranges for resistors
        self.color_ranges = {
            'black': [((0, 0, 0), (255, 50, 50))],
            'brown': [((0, 60, 20), (20, 255, 80))],
            'red': [((0, 60, 40), (10, 255, 100)), ((170, 60, 40), (180, 255, 100))],
            'orange': [((10, 60, 50), (30, 255, 120))],
            'yellow': [((25, 60, 60), (40, 255, 140))],
            'green': [((50, 60, 40), (80, 255, 100))],
            'blue': [((90, 60, 40), (120, 255, 100))],
            'violet': [((130, 60, 40), (160, 255, 100))],
            'gray': [((0, 0, 60), (255, 50, 100))],
            'white': [((0, 0, 120), (255, 50, 255))],
            'gold': [((20, 60, 80), (40, 255, 160))],
            'silver': [((0, 0, 80), (255, 50, 140))]
        }

        # Background color ranges for 4-band and 5-band resistors
        self.bc_ranges = {
            '4-band': [((0, 0, 60), (255, 50, 100))],
            '5-band': [((90, 30, 80), (120, 100, 140))]
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
        """
            Periodically updates the server with the current resistor data and total resistance.

            This method runs in a loop while the application is active. It sends the following data to the server:
            - A list of resistors with their resistance values.
            - The total resistance of the circuit.

            If an error occurs during the update, it logs the exception and continues the loop.

            Behavior:
            - Checks if the detection process is running (`self.running`).
            - Sends POST requests to update the server with resistor data and total resistance.
            - Waits for 2 seconds before the next update.

            Note:
            - This method is intended to run in a separate thread.
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
            Starts the object detection process.

            This method performs the following steps:
            - Checks if detection is already running.
            - Prompts the user to select a camera using the `select_camera` function.
            - If a valid camera is selected, sets the `running` flag to True.
            - Starts a thread for object detection (`detect_objects`).
            - Starts another thread for cleaning up old entries (`cleanup_old_entries`).
            - Displays an informational message if detection is already running.
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
            Stops the object detection process.

            This method performs the following steps:
            - Checks if the detection process is currently running.
            - If running, stops the detection by setting the `running` flag to False.
            - Waits for the detection and cleanup threads to finish using `join`.
            - Displays an informational message indicating that detection has stopped.
            - If detection is not running, displays an informational message stating so.
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
            Captures video frames from the selected camera and performs object detection.

            This method:
            - Opens the camera using the selected camera index.
            - Continuously reads frames while the detection process is running.
            - Uses the YOLO model to track objects in each frame.
            - Updates the internal data structures with detected object IDs and their last seen timestamps.
            - Calls the `recognize_resistor_colors` method to process detected objects.
            - Displays the video feed with bounding boxes and detection results in a window.
            - Stops the process if the 'q' key is pressed.

            Behavior:
            - Releases the camera and closes all OpenCV windows when the detection stops.
            - Displays an error message if the camera cannot be opened.

            Note:
            - This method is intended to run in a separate thread.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame = frame

            self.results = self.model.track(source=frame, conf=0.3, show_conf=False, persist=True,device="cpu")
            # With GPU
            # self.results = self.model.track(source=frame, conf=0.3, show_conf=False, persist=True, device="0")

            for result in self.results:
                for obj in result.boxes:
                    if obj.id is not None:
                        yolo_id = int(obj.id.item())
                        if yolo_id not in self.object_ids:
                            self.object_ids[yolo_id] = len(self.object_ids) + 1
                        self.last_seen[yolo_id] = time.time()

            self.recognize_resistor_colors(frame, self.results)
            cv2.imshow("EduVision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def mask_colors_along_lines(self, hsv_roi, lines):
        """
            Creates a mask for specific colors along detected lines in a region of interest (ROI).

            Args:
                hsv_roi (numpy.ndarray): The HSV image of the region of interest.
                lines (list): A list of lines, where each line is represented as (x1, y1, x2, y2).

            Returns:
                numpy.ndarray: A binary mask highlighting the specified colors along the detected lines.

            Behavior:
                - Initializes an empty mask with the same dimensions as the ROI.
                - Draws lines on the mask based on the provided line coordinates.
                - Iterates through predefined color ranges and applies bitwise operations to isolate colors
                  along the drawn lines.
                - Combines the results into a final mask.
        """
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                center_x1 = (x1 + x2) // 2
                center_y1 = (y1 + y2) // 2
                center_x2 = center_x1
                center_y2 = center_y1

                cv2.line(mask, (center_x1, center_y1), (center_x2, center_y2), 255, 2)

        final_mask = np.zeros_like(mask)
        for color_name, hsv_ranges in self.color_ranges.items():
            for lower, upper in hsv_ranges:
                color_mask = cv2.inRange(hsv_roi, lower, upper)
                final_mask = cv2.bitwise_or(final_mask, cv2.bitwise_and(color_mask, mask))

        return final_mask

    def rgb_to_hsl(self, rgb):
        """
            Converts an RGB color to HSL (Hue, Saturation, Lightness) format.

            Args:
                rgb (numpy.ndarray): An array containing the RGB values, where each value is in the range [0, 255].

            Returns:
                numpy.ndarray: An array containing the HSL values:
                    - Hue (H) scaled to [0, 256]
                    - Saturation (S) scaled to [0, 255]
                    - Lightness (L) scaled to [0, 255]

            Behavior:
                - Normalizes the RGB values to the range [0, 1].
                - Calculates the maximum and minimum values among R, G, and B.
                - Computes Lightness (L) as the average of the max and min values.
                - If the max and min values are equal, sets Saturation (S) and Hue (H) to 0.
                - Otherwise, calculates Saturation (S) based on the difference between max and min.
                - Determines Hue (H) based on which color channel (R, G, or B) is the maximum.
                - Scales the resulting HSL values to the appropriate ranges.
        """
        r, g, b = rgb / 255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        l = (max_val + min_val) / 2
        if max_val == min_val:
            s = h = 0
        else:
            d = max_val - min_val
            s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
            if max_val == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif max_val == g:
                h = (b - r) / d + 2
            else:
                h = (r - g) / d + 4
            h /= 6
        return np.array([h * 256, s * 255, l * 255], dtype=np.float32)

    def modified_niblack_threshold(self, gray, window_size, j=0.0036):
        """
            Applies a modified Niblack thresholding algorithm to a grayscale image.

            Args:
                gray (numpy.ndarray): The input grayscale image.
                window_size (int): The size of the sliding window used for local thresholding.
                j (float): A parameter that adjusts the standard deviation term in the threshold calculation.

            Returns:
                numpy.ndarray: A binary image where pixels are set to 255 if they are above the calculated threshold,
                               and 0 otherwise.

            Behavior:
                - Pads the input image to handle edge cases for the sliding window.
                - Iterates over each pixel in the image and calculates the local mean and standard deviation.
                - Computes a threshold value for each pixel based on the local statistics.
                - Sets the pixel value to 255 if it is greater than the threshold, otherwise sets it to 0.
        """
        h, w = gray.shape
        threshold_img = np.zeros_like(gray)
        pad = window_size // 2
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        for y in range(h):
            for x in range(w):
                window = padded[y:y + window_size, x:x + window_size]
                m = np.mean(window)
                g = window.flatten()
                n = g.size
                std_term = np.sqrt((j / n) * np.sum(g ** 2))
                T = m + std_term
                threshold_img[y, x] = 255 if gray[y, x] > T else 0
        return threshold_img

    def align_resistor(self, gray, binary):
        """
            Aligns a resistor in the image to a horizontal orientation.

            Args:
                gray (numpy.ndarray): The grayscale image of the resistor.
                binary (numpy.ndarray): The binary mask of the resistor.

            Returns:
                tuple: A tuple containing the aligned grayscale image and the rotation angle in degrees.

            Behavior:
                - Finds contours in the binary mask and selects the largest one.
                - Computes the convex hull of the largest contour.
                - Calculates image moments to determine the orientation of the resistor.
                - Rotates the image to align the resistor horizontally.
                - Returns the aligned image and the rotation angle.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray, 0
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)

        moments = cv2.moments(hull)
        if moments['m00'] == 0:
            return gray, 0
        A = moments['m00']
        cx = moments['m10'] / A
        cy = moments['m01'] / A
        Mxx = moments['m20'] / A - cx ** 2
        Myy = moments['m02'] / A - cy ** 2
        Mxy = moments['m11'] / A - cx * cy
        if abs(Mxx - Myy) < 1e-10:
            theta_rad = 0
        else:
            theta_rad = 0.5 * math.atan2(2 * Mxy, Myy - Mxx)
        theta_deg = math.degrees(theta_rad)

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), theta_deg, 1)
        aligned = cv2.warpAffine(gray, M, (w, h))
        return aligned, theta_deg

    def segment_resistor_body(self, img, binary):
        """
            Segments the body of a resistor from the given image.

            Args:
                img (numpy.ndarray): The input image of the resistor.
                binary (numpy.ndarray): The binary mask of the resistor.

            Returns:
                numpy.ndarray: The segmented region of the resistor body.

            Behavior:
                - Extracts a region of interest (ROI) around the vertical center of the image.
                - Applies erosion to the binary mask to remove noise and refine the resistor's shape.
                - Identifies the top and bottom borders of the resistor in the eroded mask.
                - Determines the left and right boundaries of the resistor body based on the borders.
                - Returns the segmented region of the resistor body.
        """
        h, w = img.shape[:2]
        center_y = h // 2
        roi = img[max(0, center_y - 40):min(h, center_y + 40), :]
        binary_roi = binary[max(0, center_y - 40):min(h, center_y + 40), :]

        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_roi, kernel, iterations=20)

        top_border = np.where(eroded[0, :])[0]
        bottom_border = np.where(eroded[-1, :])[0]
        if len(top_border) < 2 or len(bottom_border) < 2:
            return roi
        x_left = max(top_border[0], bottom_border[0])
        x_right = min(top_border[-1], bottom_border[-1])
        return img[:, x_left:x_right + 1]

    def extract_color_bands(self, img, j=0.0036):
        """
        Extracts color bands from a resistor image.

        Args:
            img (numpy.ndarray): The input image of the resistor.
            j (float): A parameter for the modified Niblack thresholding.

        Returns:
            tuple: A sorted list of color bands (each represented as a tuple of the band image and its position)
                   and a boolean indicating whether the resistor is a 4-band type.

        Behavior:
            - Calculates the background color of the image using histograms.
            - Subtracts the background color to isolate the resistor.
            - Applies modified Niblack thresholding to create a binary mask.
            - Performs connected component analysis to identify potential color bands.
            - Determines whether the resistor is a 4-band or 5-band type based on predefined background color ranges.
            - Ensures the number of detected bands matches the expected count for the resistor type.
        """
        # Find background color
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
        bc = (np.argmax(hist_r), np.argmax(hist_g), np.argmax(hist_b))

        # Generate background image and subtract
        bc_img = np.full_like(img, bc)
        diff = cv2.absdiff(img, bc_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        binary_diff = self.modified_niblack_threshold(diff_gray, window_size=95, j=j)

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_diff, connectivity=8)
        color_bands = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_HEIGHT] >= 80:  # Height threshold
                xc = int(centroids[i, 0])
                x_start = max(0, xc - 10)
                x_end = min(img.shape[1], xc + 10)
                cb = img[:, x_start:x_end]
                color_bands.append((cb, xc))

        # Determine if 4-band or 5-band
        hsl_bc = self.rgb_to_hsl(np.array(bc))
        is_4band = any(
            all(lower[i] <= hsl_bc[i] <= upper[i] for i in range(3)) for lower, upper in self.bc_ranges['4-band'])
        expected_bands = 4 if is_4band else 5
        if len(color_bands) != expected_bands:
            if is_4band and len(color_bands) != 4:
                color_bands = color_bands[:4] if len(color_bands) > 4 else color_bands
            else:
                color_bands = color_bands[:5] if len(color_bands) > 5 else color_bands
        return sorted(color_bands, key=lambda x: x[1]), is_4band

    def identify_colors(self, color_bands):
        """
        Identifies the colors of resistor bands from the given color band regions.

        Args:
            color_bands (list): A list of tuples where each tuple contains a color band image (numpy array)
                                and its position.

        Returns:
            list: A list of recognized color names corresponding to the detected color bands.

        Behavior:
            - Converts each color band image to the HLS color space.
            - Uses KMeans clustering to determine the dominant color in the band.
            - Matches the dominant color against predefined color ranges to identify the color name.
            - Appends the recognized color name to the result list if a match is found.
        """
        recognized_colors = []
        for cb, _ in color_bands:
            hsl = cv2.cvtColor(cb, cv2.COLOR_BGR2HLS)
            pixels = hsl.reshape(-1, 3)
            kmeans = KMeans(n_clusters=1, random_state=42)
            kmeans.fit(pixels)
            center = kmeans.cluster_centers_[0]
            min_dist = float('inf')
            color_name = None
            for name, ranges in self.color_ranges.items():
                for lower, upper in ranges:
                    if all(lower[i] <= center[i] <= upper[i] for i in range(3)):
                        dist = np.sum((center - (np.array(lower) + np.array(upper)) / 2) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            color_name = name
            if color_name:
                recognized_colors.append(color_name)
        return recognized_colors

    def check_sequence(self, colors, is_4band):
        """
        Ensures the correct sequence of color bands for a resistor.

        Args:
            colors (list): A list of detected color bands.
            is_4band (bool): Indicates whether the resistor is a 4-band type.

        Returns:
            list: The reordered color bands if necessary, otherwise the original list.

        Behavior:
            - Reverses the color sequence if the first or second band is gold or silver.
            - For resistors with 4 or more bands, applies additional logic to determine the correct order.
        """
        if not colors:
            return colors
        if colors[0] in ['gold', 'silver'] or (len(colors) > 1 and colors[1] in ['gold', 'silver']):
            return colors[::-1]
        if len(colors) >= 4:
            d_1L_2L = 1
            d_1R_2R = 1
            if d_1L_2L > d_1R_2R:
                return colors[::-1]
        return colors

    def recognize_resistor_colors(self, frame, results):
        """
        Recognizes resistor colors from detected objects in the frame.

        Args:
            frame (numpy.ndarray): The current video frame from the camera.
            results (list): The detection results from the YOLO model, containing bounding boxes and object IDs.

        Behavior:
            - Iterates through detected objects and extracts the region of interest (ROI) for each resistor.
            - Applies thresholding and alignment to prepare the resistor for color band extraction.
            - Segments the resistor body and identifies color bands using predefined color ranges.
            - Stabilizes color detections over multiple frames to ensure accuracy.
            - Calculates the resistance value based on the detected color bands.
            - Updates the UI with the detected resistor's ID, resistance, and bounding box.
        """
        for result in results:
            for obj in result.boxes:
                if obj.id is None:
                    continue
                yolo_id = int(obj.id.item())
                x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Threshold and align
                binary = self.modified_niblack_threshold(gray, window_size=min(roi.shape[:2]), j=0.0036)
                aligned, theta = self.align_resistor(gray, binary)
                aligned_color = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR) if len(aligned.shape) == 2 else aligned

                # Segment resistor body
                binary_aligned = self.modified_niblack_threshold(aligned, window_size=min(aligned.shape[:2]), j=0.0036)
                resistor_body = self.segment_resistor_body(aligned_color, binary_aligned)

                # Extract and identify color bands
                color_bands, is_4band = self.extract_color_bands(resistor_body)
                recognized_colors = self.identify_colors(color_bands)

                # Check sequence
                recognized_colors = self.check_sequence(recognized_colors, is_4band)

                # Store and stabilize detections
                if yolo_id not in self.color_detections:
                    self.color_detections[yolo_id] = []
                    self.resistor_frame_count[yolo_id] = 0
                if recognized_colors:
                    self.color_detections[yolo_id].append(tuple(recognized_colors))
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

                # Display results
                if yolo_id in self.resistor_last_result:
                    resistance = self.resistor_last_result[yolo_id]
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Display ID and resistance
                    cv2.putText(frame, f"ID: {yolo_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                2)
                    cv2.putText(frame, f"{resistance} Ohms", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

    def update_resistor_entry(self, yolo_id, colors, resistance, orientation, bbox):
        """
        Updates the entry for a resistor in the internal data structures and UI table.

        Args:
            yolo_id (int): The unique identifier for the resistor.
            colors (list): A list of detected color bands for the resistor.
            resistance (float): The calculated resistance value of the resistor.
            orientation (str): The orientation of the resistor (e.g., horizontal or vertical).
            bbox (tuple): The bounding box coordinates of the resistor (x1, y1, x2, y2).

        Behavior:
            - Ensures the `colors` list has exactly 4 elements by truncating or padding with "N/A".
            - Updates the `resistors_data` dictionary with the resistor's details.
            - Adds or updates the resistor's entry in the UI table.
            - Recalculates the global resistance of the circuit.
        """
        if len(colors) > 4:
            colors = colors[:4]
        elif len(colors) < 4:
            colors += ["N/A"] * (4 - len(colors))

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
        Periodically removes old entries from the UI and internal data structures.

        This method runs in a separate thread while the application is running. It checks
        for resistors that have not been updated within the last 2 seconds and removes
        their corresponding entries from the table, `resistors_data`, and `last_seen`.

        Steps:
        - Iterates through `last_seen` to find resistors that have not been updated.
        - Deletes their entries from the UI table and internal dictionaries.
        - Sleeps for 1 second before repeating the process.

        Note:
        - This method assumes `self.running` is True while the application is active.
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
        Calculates the total resistance of the circuit by determining
        whether resistors are connected in series or parallel.

        Logic:
        - Iterates through all resistors to classify them as parallel or series.
        - Computes the total resistance for series and parallel connections.
        - Updates the global resistance label in the UI.

        Notes:
        - If any resistor configuration is invalid, the total resistance is set to infinity.
        - Series resistance is the sum of individual resistances.
        - Parallel resistance is calculated as the reciprocal of the sum of reciprocals.
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
        Determines if a resistor is connected in parallel with another resistor.

        Args:
            yolo_id (int): The ID of the resistor to check.

        Returns:
            bool: True if the resistor is in parallel with another, False otherwise.

        Logic:
            - Calculates the center coordinates of the bounding box for the given resistor.
            - Iterates through all other resistors to compare their positions.
            - Checks if the distance between the resistors is below a threshold.
            - Verifies if the resistors are aligned horizontally (small vertical difference)
              and sufficiently separated horizontally.
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
        Determines if a resistor is connected in series with another resistor.

        Args:
            yolo_id (int): The ID of the resistor to check.
            parallel_resistors (dict): A dictionary of resistors identified as parallel.

        Returns:
            bool: True if the resistor is in series with another, False otherwise.

        Logic:
            - Calculates the center coordinates of the bounding box for the given resistor.
            - Iterates through all other resistors to compare their positions.
            - Checks if the distance between the resistors is below a threshold.
            - Verifies if the resistors are aligned vertically (small horizontal difference)
              and sufficiently separated vertically.
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


if __name__ == "__main__":
    """
        Initializes the EduVision application.
    """
    root = tk.Tk()
    app = EduVision(root)
    root.mainloop()