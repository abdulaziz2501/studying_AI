import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os

class CameraGUI:
    def __init__(self, main_loop_callback):
        self.root = tk.Tk()
        self.root.title("AI Camera App - Pro Control")
        self.root.geometry("800x600")
        
        self.main_loop_callback = main_loop_callback
        self.running = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Sidebar
        sidebar = ttk.Frame(self.root, padding="10")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(sidebar, text="Controls", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        self.start_btn = ttk.Button(sidebar, text="Start Camera", command=self.start)
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = ttk.Button(sidebar, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(sidebar, text="Status:").pack(anchor=tk.W)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(sidebar, textvariable=self.status_var, foreground="blue").pack(anchor=tk.W)
        
        # Video Display
        self.video_canvas = tk.Canvas(self.root, background="black")
        self.video_canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
    def start(self):
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Running")
            
            # Start the main logic in a thread
            self.thread = threading.Thread(target=self.main_loop_callback, args=(self.update_canvas,), daemon=True)
            self.thread.start()
            
    def stop(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        
    def update_canvas(self, frame):
        """Callback for the main logic to update the Tkinter canvas."""
        if not self.running: return
        
        # Convert OpenCV BGR to Tkinter Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        
        # Resize to fit canvas
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            img = img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.imgtk = imgtk  # Keep a reference
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def run(self):
        self.root.mainloop()
        self.running = False # Ensure thread stops if window closed
