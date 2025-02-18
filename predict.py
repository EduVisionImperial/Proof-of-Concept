import cv2
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk
from ultralytics import YOLO
import time
import torch

class EduVision:
    def __init__(self, root):
        self.root = root
        self.root.title("EduVision")
        self.root.geometry("600x400")
        self.model = YOLO('./validation/best.pt')
        self.running = False

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=10)

        self.table = ttk.Treeview(root, columns=("ID", "Colors", "Resistance"), show="headings")
        self.table.heading("ID", text="ID")
        self.table.heading("Colors", text="Colors")
        self.table.heading("Resistance", text="Resistance")
        self.table.pack(pady=10, fill=tk.BOTH, expand=True)

        self.color_tensors = {}

    def list_cameras(self):
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
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
        else:
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
            messagebox.showerror(title='Error', message='Could not open camera')
            return

        last_seen = {}
        object_ids = {}

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(source=frame, conf=0.3, show_conf=False, persist=True)

            current_time = time.time()
            new_object_ids = {}

            for result in results:
                for obj in result.boxes:
                    obj_id = obj.id
                    if obj_id not in object_ids:
                        object_ids[obj_id] = len(object_ids) + 1
                    new_object_ids[obj_id] = object_ids[obj_id]
                    last_seen[obj_id] = current_time
                    annotated_frame = result.plot()
                    cv2.imshow('Webcam', annotated_frame)

            for obj_id in list(last_seen.keys()):
                if current_time - last_seen[obj_id] > 3:
                    del last_seen[obj_id]
                    del object_ids[obj_id]

            self.recognize_resistor_colors(frame, results)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_resistor_colors(self, frame, results):
        color_values = {
            'black': 0,
            'brown': 1,
            'red': 2,
            'orange': 3,
            'yellow': 4,
            'green': 5,
            'blue': 6,
            'violet': 7,
            'gray': 8,
            'white': 9
        }

        for result in results:
            for obj in result.boxes:
                x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                roi = frame[y1:y2, x1:x2]

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                color_ranges = {
                    'black': ((0, 0, 0), (180, 255, 30)),
                    'brown': ((10, 100, 20), (20, 255, 200)),
                    'red': ((0, 100, 100), (10, 255, 255)),
                    'orange': ((10, 100, 100), (25, 255, 255)),
                    'yellow': ((25, 100, 100), (35, 255, 255)),
                    'green': ((35, 100, 100), (85, 255, 255)),
                    'blue': ((85, 100, 100), (125, 255, 255)),
                    'violet': ((125, 100, 100), (145, 255, 255)),
                    'gray': ((0, 0, 50), (180, 50, 200)),
                    'white': ((0, 0, 200), (180, 20, 255))
                }

                recognized_colors = []

                for color_name, (lower, upper) in color_ranges.items():
                    mask = cv2.inRange(hsv, lower, upper)
                    if cv2.countNonZero(mask) > 0 and color_name not in recognized_colors:
                        recognized_colors.append(color_name)
                        cv2.putText(frame, color_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                if len(recognized_colors) >= 3:
                    resistance = self.calculate_resistance(recognized_colors[:3], color_values)
                    cv2.putText(frame, f"{resistance} Ohms", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    self.update_table(obj.id, recognized_colors, resistance)

                    color_tensor = torch.tensor([color_values[color] for color in recognized_colors[:3]])
                    self.color_tensors[obj.id] = color_tensor

        cv2.imshow('Resistor Colors', frame)

    def update_table(self, obj_id, colors, resistance):
        for i in self.table.get_children():
            self.table.delete(i)
        self.table.insert("", "end", values=(obj_id, ", ".join(colors), f"{resistance} Ohms"))

    def calculate_resistance(self, colors, color_values):
        first_digit = color_values[colors[0]]
        second_digit = color_values[colors[1]]
        multiplier = 10 ** color_values[colors[2]]
        return (first_digit * 10 + second_digit) * multiplier

if __name__ == "__main__":
    root = tk.Tk()
    app = EduVision(root)
    root.mainloop()