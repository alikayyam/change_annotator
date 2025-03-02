import tkinter as tk
from PIL import Image, ImageTk

class ImagePanner(tk.Tk):
    def __init__(self, image_path):
        super().__init__()
        self.title("Image Panner")
        
        # Load the image
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Create a canvas and add the image
        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Bind mouse events for panning
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        
        self.last_x = 0
        self.last_y = 0
    
    def start_pan(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def do_pan(self, event):
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.canvas.move(tk.ALL, dx, dy)
        self.last_x = event.x
        self.last_y = event.y

if __name__ == "__main__":
    app = ImagePanner(r".\data\51048_14_F2_RE_RS_51048_10_F2_RE_LS\51048_10_F2_RE_LS.jpg")
    app.mainloop()
