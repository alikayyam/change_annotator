# import tkinter as tk
# from PIL import Image, ImageTk

# class ZoomImageApp:
#     def __init__(self, root, image_path):
#         self.root = root
#         self.root.title("Zoom Image Example")

#         self.canvas = tk.Canvas(root, width=800, height=600)
#         self.canvas.pack(fill=tk.BOTH, expand=True)

#         # Load the image using PIL
#         self.original_image = Image.open(image_path)
#         self.image = self.original_image
#         self.photo_image = ImageTk.PhotoImage(self.image)

#         # Add the image to the canvas
#         self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

#         # Configure canvas scroll region
#         self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

#         # Bind the mouse wheel event
#         self.canvas.bind("<MouseWheel>", self.zoom)

#     def zoom(self, event):
#         # Get the mouse position
#         x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

#         # Calculate the zoom factor (scale up or down based on the event)
#         if event.delta > 0:
#             scale_factor = 1.1
#         else:
#             scale_factor = 0.9

#         # Resize the image
#         self.image = self.image.resize((int(self.image.width * scale_factor), int(self.image.height * scale_factor)), Image.LANCZOS)
#         self.photo_image = ImageTk.PhotoImage(self.image)

#         # Update the image on the canvas
#         self.canvas.itemconfig(self.image_id, image=self.photo_image)

#         # Adjust the scroll region to the new image size
#         self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

#         # Calculate the new position to keep the zoom centered on the mouse position
#         new_x = scale_factor * x - x
#         new_y = scale_factor * y - y
#         self.canvas.move(self.image_id, -new_x, -new_y)

#         # Update the canvas view to keep the mouse position stable
#         self.canvas.xview_scroll(int(new_x), 'units')
#         self.canvas.yview_scroll(int(new_y), 'units')

# if __name__ == "__main__":
#     root = tk.Tk()

#     app = ZoomImageApp(root, r".\data\51048_14_F2_RE_RS_51048_10_F2_RE_LS\51048_10_F2_RE_LS.jpg")
#     root.mainloop()





# import tkinter as tk
# from PIL import Image, ImageTk

# class ZoomApp:
#     def __init__(self, root, image_path):
#         self.root = root
#         self.canvas = tk.Canvas(root, width=800, height=600)
#         self.canvas.pack(fill="both", expand=True)
        
#         self.image = Image.open(image_path)
#         self.tk_image = ImageTk.PhotoImage(self.image)
#         self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
#         self.canvas.bind("<MouseWheel>", self.zoom)
        
#         self.scale_factor = 1.1  # Zoom factor

#     def zoom(self, event):
#         # Get the current mouse position
#         x, y = event.x, event.y
        
#         # Get the current canvas size and image size
#         canvas_width = self.canvas.winfo_width()
#         canvas_height = self.canvas.winfo_height()
#         img_width, img_height = self.image.size
        
#         # Calculate the zoom direction and factor
#         if event.delta > 0:  # Zoom in
#             factor = self.scale_factor
#         else:  # Zoom out
#             factor = 1 / self.scale_factor
        
#         # Calculate new size
#         new_width = int(img_width * factor)
#         new_height = int(img_height * factor)
        
#         # Resize image
#         self.image = self.image.resize((new_width, new_height))#, Image.ANTIALIAS)
#         self.tk_image = ImageTk.PhotoImage(self.image)
        
#         # Calculate new position to keep the mouse position constant
#         mouse_x_ratio = x / canvas_width
#         mouse_y_ratio = y / canvas_height
        
#         new_x = x - mouse_x_ratio * new_width
#         new_y = y - mouse_y_ratio * new_height
        
#         # Update image on canvas
#         self.canvas.delete(self.image_id)
#         self.image_id = self.canvas.create_image(new_x, new_y, anchor="nw", image=self.tk_image)

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ZoomApp(root, r".\data\51048_14_F2_RE_RS_51048_10_F2_RE_LS\51048_10_F2_RE_LS.jpg")
#     root.mainloop()



import tkinter as tk
from PIL import Image, ImageTk

class ZoomApp:
    def __init__(self, root, image_path):
        self.root = root
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
        self.canvas.bind("<MouseWheel>", self.zoom)
        
        self.scale_factor = 1.1  # Zoom factor


        # Label to display the mouse position
        self.position_label = tk.Label(self.root)
        self.position_label.pack(side=tk.RIGHT, padx=(0, 15))


    def zoom(self, event):
        # Get the current mouse position
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Get the current image size
        img_width, img_height = self.image.size
        
        # Calculate the zoom direction and factor
        if event.delta > 0:  # Zoom in
            factor = self.scale_factor
        else:  # Zoom out
            factor = 1 / self.scale_factor
        
        # Calculate new size
        new_width = int(img_width * factor)
        new_height = int(img_height * factor)
        
        # Resize image
        self.image = self.image.resize((new_width, new_height)) #, Image.Resampling.BICUBIC)
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Calculate new position to keep the mouse position constant
        new_x = x - factor * (x - self.canvas.coords(self.image_id)[0])
        new_y = y - factor * (y - self.canvas.coords(self.image_id)[1])
        
        # Update image on canvas
        self.canvas.delete(self.image_id)
        self.image_id = self.canvas.create_image(new_x, new_y, anchor="nw", image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ZoomApp(root, r".\data\51048_14_F2_RE_RS_51048_10_F2_RE_LS\51048_10_F2_RE_LS.jpg")
    root.mainloop()
