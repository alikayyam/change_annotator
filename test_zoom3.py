import tkinter as tk
from PIL import Image, ImageTk

class ZoomPanCanvas:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=800, height=600, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        
        # Load an image
        self.current_image = Image.open(r".\data\51048_14_F2_RE_RS_51048_10_F2_RE_LS\51048_10_F2_RE_LS.jpg")
        self.photo = ImageTk.PhotoImage(self.current_image)
        
        # Create an image on the canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

        # Initial values
        self.zoom_factor = 1.0
        self.scale_factor = 1.1  # Zoom factor per scroll step
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        # Bind events
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)

    def zoom(self, event):
        img_width, img_height = self.current_image.size
        x, y = event.x, event.y

        # Calculate the zoom direction and factor
        if event.delta > 0:  # Zoom in
            scale = self.scale_factor
        else:  # Zoom out
            scale = 1 / self.scale_factor

        # New zoom factor
        new_zoom_factor = self.zoom_factor * scale

        # Calculate new size
        new_width = int(img_width * new_zoom_factor)
        new_height = int(img_height * new_zoom_factor)

        # Resize image
        tmp_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(tmp_image)

        # Calculate the new offset to keep the zoom centered on the mouse cursor
        offset_x = (x - self.pan_offset_x) * (new_zoom_factor / self.zoom_factor)
        offset_y = (y - self.pan_offset_y) * (new_zoom_factor / self.zoom_factor)

        self.pan_offset_x = x - offset_x
        self.pan_offset_y = y - offset_y
        self.zoom_factor = new_zoom_factor

        # Update image on canvas
        self.canvas.delete(self.image_on_canvas)
        self.image_on_canvas = self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor='nw', image=self.photo)

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

# Create the main window
root = tk.Tk()
app = ZoomPanCanvas(root)
root.mainloop()
