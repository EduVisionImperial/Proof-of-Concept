import cv2
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from ultralytics import YOLO
import time

class EduVision:
    def __init__(self, root):
        self.root = root
        self.root.title("EduVision")
        self.root.geometry("600x400")

        self.model = YOLO("./validation/best.pt")
        self.running = False
        self.resistors_data = {}
        self.tree_items = {}
        self.object_ids = {}
        self.frozen_objects = {}
        self.results = []

        self.color_values = {
            "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
            "green": 5, "blue": 6, "violet": 7, "gray": 8, "white": 9, "gold": 10, "silver": 11
        }

        self.color_ranges = {
            "black": ((0, 0, 0), (180, 255, 30)),
            "brown": ((10, 100, 20), (20, 255, 200)),
            "red": ((0, 100, 100), (10, 255, 255)),
            "orange": ((10, 100, 100), (25, 255, 255)),
            "yellow": ((25, 100, 100), (35, 255, 255)),
            "green": ((35, 100, 100), (85, 255, 255)),
            "blue": ((85, 100, 100), (125, 255, 255)),
            "violet": ((125, 100, 100), (145, 255, 255)),
            "gray": ((0, 0, 50), (180, 50, 200)),
            "white": ((0, 0, 200), (180, 20, 255)),
            "gold": ((20, 100, 100), (30, 255, 255)),
            "silver": ((0, 0, 192), (180, 50, 255))
        }

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=5)

        self.table = ttk.Treeview(root, columns=("ID", "Colors", "Resistance", "Action"), show="headings")
        self.table.heading("ID", text="ID")
        self.table.heading("Colors", text="Colors")
        self.table.heading("Resistance", text="Resistance")
        self.table.heading("Action", text="Action")
        self.table.pack(pady=5, fill=tk.BOTH, expand=True)

        self.table.bind("<Double-1>", self.on_row_double_click)

    def list_cameras(self):
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            available_cameras.append(index)
            cap.release()
            index += 1
        return available_cameras

    def select_camera(self):
        cameras = self.list_cameras()
        if not cameras:
            messagebox.showerror("Error", "No cameras found")
            return None
        camera_index = simpledialog.askinteger("Select Camera", f"Available cameras: {cameras}\nEnter camera index:")
        if camera_index in cameras:
            return camera_index
        messagebox.showerror("Error", "Invalid camera index")
        return None

    def start_detection(self):
        if not self.running:
            self.camera_index = self.select_camera()
            if self.camera_index is not None:
                self.running = True
                self.thread = threading.Thread(target=self.detect_objects)
                self.thread.start()
        else:
            messagebox.showinfo("Info", "Detection is already running")

    def stop_detection(self):
        if self.running:
            self.running = False
            self.thread.join()
            messagebox.showinfo("Info", "Detection stopped")
        else:
            messagebox.showinfo("Info", "Detection is not running")

    def detect_objects(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        last_seen = {}

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame = frame

            self.results = self.model.track(source=frame, conf=0.3, show_conf=False, persist=True)
            current_time = time.time()

            for result in self.results:
                for obj in result.boxes:
                    if obj.id is not None:
                        yolo_id = int(obj.id.item())
                        if yolo_id not in self.object_ids:
                            self.object_ids[yolo_id] = len(self.object_ids) + 1
                        last_seen[yolo_id] = current_time

            for yolo_id in list(last_seen.keys()):
                if current_time - last_seen[yolo_id] > 3:
                    del last_seen[yolo_id]
                    if yolo_id in self.resistors_data:
                        del self.resistors_data[yolo_id]
                    if yolo_id in self.tree_items:
                        self.table.delete(self.tree_items[yolo_id])
                        del self.tree_items[yolo_id]

            self.recognize_resistor_colors(frame, self.results)
            cv2.imshow("EduVision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_resistor_colors(self, frame, results):
        for result in results:
            for obj in result.boxes:
                if obj.id is not None:
                    yolo_id = int(obj.id.item())
                    x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())

                    if yolo_id in self.frozen_objects and self.frozen_objects[yolo_id]:
                        resistance = self.table.item(self.tree_items[yolo_id], "values")[2]
                        cv2.putText(frame, f"ID: {yolo_id}", (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, resistance, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        continue

                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi = frame[y1:y2, x1:x2]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    recognized_colors = []

                    for color_name, (lower, upper) in self.color_ranges.items():
                        mask = cv2.inRange(hsv, lower, upper)
                        if cv2.countNonZero(mask) > 0 and color_name not in recognized_colors:
                            recognized_colors.append(color_name)

                    cv2.putText(frame, f"ID: {yolo_id}", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    if len(recognized_colors) >= 4:
                        resistance, tolerance = self.calculate_resistance(recognized_colors[:4], self.color_values)
                        cv2.putText(frame, f"{resistance} Ohms {tolerance}%", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        self.update_resistor_entry(yolo_id, recognized_colors[:4], resistance, tolerance)

    def calculate_resistance(self, colors, color_values):
        first_digit = color_values[colors[0]]
        second_digit = color_values[colors[1]]
        multiplier = 10 ** color_values[colors[2]]
        resistance = (first_digit * 10 + second_digit) * multiplier

        tolerance_values = {
            "brown": 1, "red": 2, "green": 0.5, "blue": 0.25, "violet": 0.1, "gray": 0.05, "gold": 5, "silver": 10
        }

        tolerance = tolerance_values.get(colors[3], 0)
        return resistance, tolerance

    def update_resistor_entry(self, yolo_id, colors, resistance, tolerance):
        self.resistors_data[yolo_id] = {
            "colors": colors,
            "resistance": resistance,
            "tolerance": tolerance
        }
        res_str = f"{resistance} Ohms {tolerance}%"

        if yolo_id not in self.tree_items:
            row_id = self.table.insert("", "end", values=(yolo_id, colors, res_str, "Freeze"))
            self.tree_items[yolo_id] = row_id
            self.table.set(row_id, column="Action", value="Freeze")
        else:
            self.table.item(self.tree_items[yolo_id], values=(yolo_id, colors, res_str, "Freeze"))

    def on_row_double_click(self, event):
        item_id = self.table.selection()[0]
        yolo_id = int(self.table.item(item_id, "values")[0])
        colors = self.resistors_data[yolo_id]["colors"]

        new_colors = simpledialog.askstring("Rearrange Colors",
                                            f"Current colors: {', '.join(colors)}\nEnter new colors (comma-separated):")
        if new_colors:
            new_colors_list = [color.strip() for color in new_colors.split(",")]
            if len(new_colors_list) >= 4:
                if yolo_id not in self.frozen_objects or not self.frozen_objects[yolo_id]:
                    resistance, tolerance = self.calculate_resistance(new_colors_list[:4], self.color_values)
                    self.update_resistor_entry(yolo_id, new_colors_list[:4], resistance, tolerance)
                    self.update_frame_with_new_colors(self.frame, yolo_id, new_colors_list[:4])
                self.frozen_objects[yolo_id] = True
                self.table.set(self.tree_items[yolo_id], column="Action", value="Unfreeze")

    def update_frame_with_new_colors(self, frame, yolo_id, new_colors):
        for result in self.results:
            for obj in result.boxes:
                if obj.id is not None and int(obj.id.item()) == yolo_id:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                    roi = frame[y1:y2, x1:x2]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    frame[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    for color_name in new_colors:
                        lower, upper = self.color_ranges[color_name]
                        mask = cv2.inRange(hsv, lower, upper)
                        if cv2.countNonZero(mask) > 0:
                            cv2.putText(frame, color_name, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    if yolo_id not in self.frozen_objects or not self.frozen_objects[yolo_id]:
                        resistance, tolerance = self.calculate_resistance(new_colors[:4], self.color_values)
                        cv2.putText(frame, f"{resistance} Ohms {tolerance}%", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    break

    def toggle_freeze(self, yolo_id):
        if yolo_id in self.frozen_objects and self.frozen_objects[yolo_id]:
            self.frozen_objects[yolo_id] = False
            self.table.set(self.tree_items[yolo_id], column="Action", value="Freeze")
        else:
            self.frozen_objects[yolo_id] = True
            self.table.set(self.tree_items[yolo_id], column="Action", value="Unfreeze")

    def on_freeze_button_click(self, event):
        item_id = self.table.selection()[0]
        yolo_id = int(self.table.item(item_id, "values")[0])
        self.toggle_freeze(yolo_id)

if __name__ == "__main__":
    root = tk.Tk()
    app = EduVision(root)
    root.mainloop()