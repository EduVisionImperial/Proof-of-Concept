import cv2
import os
import time
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog

class CaptureImagesFromCamera:
    def __init__(self, root):
        self.root = root
        self.root.title("EduVision - Capture Images")
        self.root.geometry("300x200")

        self.prefix_label = tk.Label(root, text="Image Prefix:")
        self.prefix_label.pack(pady=5)

        self.prefix_entry = tk.Entry(root)
        self.prefix_entry.pack(pady=5)

        self.start_button = tk.Button(root, text="Start Capturing", command=self.start_capturing)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Capturing", command=self.stop_capturing)
        self.stop_button.pack(pady=10)

        self.capturing = False
        self.capture_thread = None
        self.camera_index = None

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

    def create_images_folder(self):
        if not os.path.exists('images'):
            os.makedirs('images')

    def capture_photos(self):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return

        self.create_images_folder()
        img_counter = 0
        prefix = self.prefix_entry.get()

        try:
            while self.capturing:
                ret, frame = cap.read()

                if not ret:
                    messagebox.showerror("Error", "Could not read frame.")
                    break

                img_name = f"images/{prefix}_{img_counter:04d}.png"
                cv2.imwrite(img_name, frame)
                print(f"{img_name} written!")

                img_counter += 1
                time.sleep(3)

        except Exception as e:
            messagebox.showerror("Error", str(e))

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def start_capturing(self):
        if not self.capturing:
            self.camera_index = self.select_camera()
            if self.camera_index is not None:
                self.capturing = True
                self.capture_thread = threading.Thread(target=self.capture_photos)
                self.capture_thread.start()
        else:
            messagebox.showinfo("Info", "Capturing is already running")

    def stop_capturing(self):
        if self.capturing:
            self.capturing = False
            self.capture_thread.join()
            messagebox.showinfo("Info", "Capturing stopped")
        else:
            messagebox.showinfo("Info", "Capturing is not running")

if __name__ == "__main__":
    root = tk.Tk()
    app = CaptureImagesFromCamera(root)
    root.mainloop()