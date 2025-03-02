import tkinter as tk
from PIL import Image, ImageTk

class ImagePanner(tk.Tk):
    def __init__(self, image_path):
        super().__init__()
        self.title("Image Panner")
        
        # Load the image
        self.image = Image.open(image_path)
        self.original_image = self.image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Create a canvas and add the image
        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Bind mouse events for panning and zooming
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Motion>", self.show_mouse_position)
        
        # Add Home button with specified width
        self.home_button = tk.Button(self, text="Home", command=self.reset_view, width=10)
        self.home_button.pack(side=tk.BOTTOM)
        
        # Bind Ctrl+Home key combination
        self.bind_all("<Control-Home>", self.reset_view)
        
        self.last_x = 0
        self.last_y = 0
        self.scale_factor = 1.0
        
        # Label to display the mouse position
        self.position_label = tk.Label(self, text="Mouse Position: ")
        self.position_label.pack(side=tk.BOTTOM)
        
        # Add radio buttons for selection
        self.radio_var = tk.StringVar(value="Option1")  # Initial value
        
        self.radio_frame = tk.Frame(self)
        self.radio_frame.pack(side=tk.BOTTOM, pady=10)
        
        self.radio_button1 = tk.Radiobutton(self.radio_frame, text="Option 1", variable=self.radio_var, value="Option1")
        self.radio_button2 = tk.Radiobutton(self.radio_frame, text="Option 2", variable=self.radio_var, value="Option2")
        self.radio_button3 = tk.Radiobutton(self.radio_frame, text="Option 3", variable=self.radio_var, value="Option3")
        
        self.radio_button1.pack(side=tk.LEFT)
        self.radio_button2.pack(side=tk.LEFT)
        self.radio_button3.pack(side=tk.LEFT)
    
    def start_pan(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def do_pan(self, event):
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.canvas.move(tk.ALL, dx, dy)
        self.last_x = event.x
        self.last_y = event.y
    
    def zoom(self, event):
        # Calculate the new scale factor
        if event.delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        
        # Resize the image
        new_width = int(self.original_image.width * self.scale_factor)
        new_height = int(self.original_image.height * self.scale_factor)
        self.image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Update the canvas image
        self.canvas.itemconfig(self.image_id, image=self.tk_image)
        
        # Center the image
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas.coords(self.image_id, (canvas_width - new_width) // 2, (canvas_height - new_height) // 2)
    
    def reset_view(self, event=None):
        self.scale_factor = 1.0
        self.image = self.original_image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_id, image=self.tk_image)
        self.canvas.coords(self.image_id, 0, 0)  # Reset image position to (0, 0)
    
    def show_mouse_position(self, event):
        x, y = event.x, event.y
        self.position_label.config(text=f"Mouse Position: {x}, {y}")

if __name__ == "__main__":
    app = ImagePanner(r".\data\51048_14_F2_RE_RS_51048_10_F2_RE_LS\51048_10_F2_RE_LS.jpg")
    app.mainloop()
