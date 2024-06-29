import cv2
import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import time
import json

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.root.geometry("1400x800")
        self.root.configure(bg="#2e2e2e")

        self.setup_ui()

        self.cap = None
        self.running = False
        self.recording = False
        self.detection_enabled = True
        self.night_mode = False
        self.out = None
        self.start_time = None
        self.recording_label = None
        self.timer_label = None

        self.load_model()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#2e2e2e")
        main_frame.pack(fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(main_frame, bg="#2e2e2e")
        button_frame.pack(pady=10)

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="groove",
                        background="#4d4d4d", foreground="black", font=("Book Antiqua", 12))
        style.map("TButton", background=[('active', '#666666')],
                  foreground=[('active', 'blue')])

        self.start_btn = ttk.Button(button_frame, text="Start Camera", command=self.start_camera, style="TButton")
        self.start_btn.grid(row=0, column=0, padx=10, pady=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop Camera", command=self.stop_camera, style="TButton")
        self.stop_btn.grid(row=1, column=0, padx=10, pady=5)

        self.screenshot_btn = ttk.Button(button_frame, text="Take Screenshot", command=self.take_screenshot, style="TButton")
        self.screenshot_btn.grid(row=0, column=1, padx=10, pady=5)

        self.toggle_btn = ttk.Button(button_frame, text="Toggle Detection", command=self.toggle_detection, style="TButton")
        self.toggle_btn.grid(row=1, column=1, padx=10, pady=5)

        self.record_btn = ttk.Button(button_frame, text=" Start Recording ", command=self.toggle_recording, style="TButton")
        self.record_btn.grid(row=0, column=2, padx=10, pady=5)

        self.night_mode_btn = ttk.Button(button_frame, text="    Night Mode    ", command=self.toggle_night_mode, style="TButton")
        self.night_mode_btn.grid(row=1, column=2, padx=10, pady=5)

        self.canvas = tk.Canvas(main_frame, width=640, height=480, bg="#2e2e2e")
        self.canvas.pack(pady=10)

        stats_frame = tk.Frame(main_frame, bg="#2e2e2e")
        stats_frame.pack(pady=10)

        self.stats_label = tk.Label(stats_frame, text="Stats: FPS = 0, Objects Detected = 0", bg="#2e2e2e", fg="green", font=("Courier New", 12))
        self.stats_label.pack()

        self.confidence_label = tk.Label(stats_frame, text="Confidence Threshold:", bg="#2e2e2e", fg="violet", font=("Courier New", 12))
        self.confidence_label.pack(side=tk.LEFT, padx=10)

        self.confidence_scale = tk.Scale(stats_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, bg="#2e2e2e", fg="red", highlightbackground="#2e2e2e", troughcolor="#4d4d4d")
        self.confidence_scale.set(0.5)
        self.confidence_scale.pack(side=tk.LEFT, padx=10)

        self.camera_label = tk.Label(stats_frame, text="Camera Index:", bg="#2e2e2e", fg="violet", font=("Courier New", 12))
        self.camera_label.pack(side=tk.LEFT, padx=10)

        self.camera_index = tk.Entry(stats_frame, font=("Courier New", 12), bg="#4d4d4d", fg="white", insertbackground="white")
        self.camera_index.pack(side=tk.LEFT, padx=10)
        self.camera_index.insert(0, "0")

    def load_model(self):
        # Paths to files
        self.image_path = 'Lenna_(test_image).png'
        self.classFile = 'coco.names'
        self.configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weightsPath = 'frozen_inference_graph.pb'

        # Check if files exist
        for file_path in [self.image_path, self.classFile, self.configPath, self.weightsPath]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Failed to open image file: {self.image_path}")

        # Load class names
        with open(self.classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        # Load model
        self.net = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def start_camera(self):
        if self.running:
            return
        camera_index = int(self.camera_index.get())
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Camera could not be opened.")
            return
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.running = True
        self.update_frame()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
        if self.recording:
            self.toggle_recording()
        self.running = False

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled

    def toggle_night_mode(self):
        self.night_mode = not self.night_mode

    def take_screenshot(self):
        if self.cap and self.running:
            success, img = self.cap.read()
            if success:
                filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
                if filename:
                    cv2.imwrite(filename, img)
                    messagebox.showinfo("Screenshot", f"Screenshot saved as {filename}")

    def toggle_recording(self):
        if not self.running:
            return
        self.recording = not self.recording
        if self.recording:
            filename = filedialog.asksaveasfilename(defaultextension=".avi",
                                                    filetypes=[("AVI files", "*.avi"), ("All files", "*.*")])
            if filename:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
                self.start_time = time.time()
                self.recording_label = tk.Label(self.root, text="Recording...", bg="#2e2e2e", fg="red", font=("Courier New", 12))
                self.recording_label.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
                self.update_timer()
                self.record_btn.config(text="Stop Recording")
            else:
                self.recording = False
        else:
            self.out.release()
            self.record_btn.config(text="Start Recording")
            if self.recording_label:
                self.recording_label.destroy()
                if self.timer_label:
                    self.timer_label.destroy()
            messagebox.showinfo("Recording", f"Video recorded successfully for {int(time.time() - self.start_time)} seconds.")

    def update_timer(self):
        if self.recording:
            elapsed_time = time.time() - self.start_time
            time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            if self.timer_label:
                self.timer_label.config(text=time_str)
            else:
                self.timer_label = tk.Label(self.root, text=time_str, bg="#2e2e2e", fg="white", font=("Courier New", 12))
                self.timer_label.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
            self.root.after(1000, self.update_timer)

    def update_frame(self):
        if not self.running:
            return

        start_time = time.time()
        success, img = self.cap.read()
        if not success:
            messagebox.showerror("Error", "Failed to read from camera.")
            self.stop_camera()
            return

        objects_detected = 0
        if self.detection_enabled:
            classIds, confs, bbox = self.net.detect(img, confThreshold=self.confidence_scale.get())
            objects_detected = len(classIds)
            if objects_detected > 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, self.classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if self.night_mode:
            img = cv2.bitwise_not(img)  # Invert colors for night mode

        if self.recording and self.out:
            self.out.write(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.root.img = img

        fps = 1.0 / (time.time() - start_time)
        self.stats_label.config(text=f"Stats: FPS = {fps:.2f}, Objects Detected = {objects_detected}")

        self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
