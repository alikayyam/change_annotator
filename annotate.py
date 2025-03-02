
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import json
from pathlib import Path
import os

class ImageToggleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Change Annotator")
        self.root.state('zoomed')

        # Initialize images
        self.image_a = None
        self.image_b = None
        self.current_image = None

        self.current_path = None

        self.image_path_1 = None
        self.image_path_2 = None

        # Initialize zoom factor
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0



        self.new_img = 0

        # Create a frame for the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Create the upload buttons
        self.upload_button_a = tk.Button(self.button_frame, text="Upload Image A", command=self.upload_image_a)
        # self.upload_button_a.place(x=10, y=10)
        self.upload_button_a.pack(side=tk.LEFT, padx=5, pady=10)

        self.upload_button_b = tk.Button(self.button_frame, text="Upload Image B", command=self.upload_image_b)
        self.upload_button_b.pack(side=tk.LEFT, padx=5, pady=10)

        # Create the toggle button
        self.toggle_button = tk.Button(self.button_frame, text="Toggle Image", command=self.toggle_image)
        self.toggle_button.pack(side=tk.LEFT, padx=55, pady=10)

        # Create the clear button
        self.clear_button = tk.Button(self.button_frame, text="Delete Boxes", command=self.clear_boxes)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=10)


        # Create the zoom in button
        self.zoom_in_button = tk.Button(self.button_frame, text="Zoom +", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.RIGHT, padx=5, pady=10)

        # Create the zoom reset
        self.zoom_reset_button = tk.Button(self.button_frame, text="Reset", command=self.zoom_reset)
        self.zoom_reset_button.pack(side=tk.RIGHT, padx=5, pady=10)

        # Create the zoom out button
        self.zoom_out_button = tk.Button(self.button_frame, text="Zoom -", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.RIGHT, padx=5, pady=10)

        # Create the save button
        self.save_button = tk.Button(self.button_frame, text="Save Boxes", command=self.save_boxes)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=10)

        # Create the upload boxes button
        self.upload_boxes_button = tk.Button(self.button_frame, text="Upload Boxes", command=self.upload_boxes)
        self.upload_boxes_button.pack(side=tk.LEFT, padx=5, pady=10)


        # # Create buttons to hide and unhide rectangles
        # self.hide_button = tk.Button(root, text="Hide Rectangles", command=self.hide_rects)
        # self.hide_button.pack()

        # self.unhide_button = tk.Button(root, text="Unhide Rectangles", command=self.unhide_rects)
        # self.unhide_button.pack()


        # Create a label to display the selected file name
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(side=tk.TOP, fill=tk.X)


        self.label_01 = tk.Label(self.label_frame, text="Image A:", wraplength=500,  fg="black")
        self.label_01.pack(side=tk.LEFT,padx=5,pady=10)

        self.label_1 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black")
        self.label_1.pack(side=tk.LEFT,padx=2,pady=10)


        self.label_02 = tk.Label(self.label_frame, text="Image B:", wraplength=500,  fg="black")
        self.label_02.pack(side=tk.LEFT,padx=5,pady=10)

        self.label_2 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black")
        self.label_2.pack(side=tk.LEFT,padx=2,pady=10)


        # Create a frame for the canvas
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create the canvas
        self.canvas = Canvas(self.canvas_frame, background="black", width = 3200, height = 2800)

        # self.canvas = tk.Canvas(self.canvas_frame, background="black", width = width,height = height)
        self.canvas.pack()

        # Bind mouse events for drawing bounding boxes
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        # Bind key event for toggling image
        self.root.bind("<space>", self.toggle_image)
        self.root.bind("<Shift-+>", self.zoom_in)
        self.root.bind("<Shift-_>", self.zoom_out)        
        self.root.bind("<Shift-)>", self.zoom_reset)                

        self.root.bind("<Shift-H>", self.hide_rects)                
        self.root.bind("<Shift-U>", self.unhide_rects)                        


        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)    
        self.canvas.bind("<Button-3>", self.delete_box)


        # Initialize drawing state
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.rectangles = []

        self.clear_boxes()


    def upload_image_a(self):

        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if file_path:
            self.image_a = Image.open(file_path)
            self.clear_boxes()  # Clear existing rectangles
            self.current_image = self.image_a
            self.current_path = file_path
            
            self.image_path_1 = file_path

            if self.image_path_1:
                self.label_1.config(text=f"{Path(self.image_path_1).name}", fg="blue")
            if self.image_path_2:    
                self.label_2.config(text=f"{Path(self.image_path_2).name}", fg="black")

            self.new_img = 1
            self.update_image()


    def upload_image_b(self):

        # self.zoom_factor = 1.0
        # self.pan_offset_x = 0
        # self.pan_offset_y = 0

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if file_path:
            self.image_b = Image.open(file_path)
            # if self.current_image is None:
            self.clear_boxes()  # Clear existing rectangles
            self.current_image = self.image_b
            self.current_path = file_path

            self.image_path_2 = file_path

            if self.image_path_1:
                self.label_1.config(text=f"{Path(self.image_path_1).name}", fg="black")
            if self.image_path_2:            
                self.label_2.config(text=f"{Path(self.image_path_2).name}", fg="blue")

            # self.new_img = 1
            self.update_image()


    def toggle_image(self, event=None):
        if self.image_a and self.image_b:
            # self.current_image = self.image_b if self.current_image == self.image_a else self.image_a

            if self.current_image == self.image_a:
                self.current_image = self.image_b
                self.label_2.config(fg="blue")
                self.label_1.config(fg="black")

            else:
                self.current_image = self.image_a
                self.label_1.config(fg="blue")
                self.label_2.config(fg="black")


            self.update_image()

    def clear_canvas(self):
        self.canvas.delete("rects")  # Delete all items on the canvas            


    def update_image(self):
        if self.current_image:
            zoomed_width = int(self.current_image.width * self.zoom_factor)
            zoomed_height = int(self.current_image.height * self.zoom_factor)
            self.photo = ImageTk.PhotoImage(self.current_image.resize((zoomed_width, zoomed_height)))
            
            if self.new_img: 
                self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.new_img = 0
            else:     
                if hasattr(self, 'image_on_canvas'):
                    self.canvas.itemconfig(self.image_on_canvas, image=self.photo)

            self.canvas.config(width=zoomed_width, height=zoomed_height)

            # Clear canvas and redraw all rectangles on the new image
            self.clear_canvas()

            print(self.zoom_factor, self.pan_offset_x, self.pan_offset_y)
            for rect in self.rectangles:
                self.canvas.create_rectangle(*self.scale_coordinates(rect), outline='yellow', tags='rects')

    def scale_coordinates(self, coords):
        return  [coord * self.zoom_factor + self.pan_offset_x if i % 2 == 0 else coord * self.zoom_factor + self.pan_offset_y for i, coord in enumerate(coords)]


    def start_draw(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='yellow', tags='rects')

    def draw(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def end_draw(self, event):
        self.rectangles.append([(t - self.pan_offset_x)/ self.zoom_factor if i % 2 == 0 else (t - self.pan_offset_y)/ self.zoom_factor for i, t in enumerate(self.canvas.coords(self.rect))])
        self.rect = None

    def clear_boxes(self):
        self.clear_canvas()
        self.rectangles.clear()


    def hide_rects(self, event=None):
        # Hide all rectangles
        for rect in self.rectangles:
            self.canvas.itemconfig(rect, state='hidden')#state=tk.HIDDEN)
        self.clear_canvas()


    def unhide_rects(self, event=None):
        # Unhide all rectangles
        for rect in self.rectangles:
            self.canvas.itemconfig(rect, state=tk.NORMAL)
            
        self.update_image()


    def zoom_in(self, event=None):
        self.zoom_factor *= 1.1
        self.update_image()

    def zoom_out(self, event=None):
        self.zoom_factor /= 1.1
        self.update_image()

    def zoom_reset(self, event=None):
        self.zoom_factor = 1.0
        self.center_image()
        self.update_image()


    def mouse_wheel(self, event):
        if (event.delta < 0):
            self.zoom_factor /= 1.1
        else:
            self.zoom_factor *= 1.1

        self.update_image()


    def start_pan(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def pan(self, event):
        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.canvas.move(self.image_on_canvas, dx, dy)
        self.update_image()


    def end_pan(self, event):
        pass
        

    def center_image(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = self.current_image.width * self.zoom_factor
        image_height = self.current_image.height * self.zoom_factor
        self.pan_offset_x = 0 #(canvas_width - image_width) / 2
        self.pan_offset_y = 0 #(canvas_height - image_height) / 2
        self.canvas.coords(self.image_on_canvas, self.pan_offset_x, self.pan_offset_y)


    def delete_box(self, event):
        for rect in self.rectangles:
            x1, y1, x2, y2 = self.scale_coordinates(rect)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                # self.canvas.delete(self.canvas.find_closest((x1 + x2) / 2, (y1 + y2) / 2))
                self.rectangles.remove(rect)
                break

        self.update_image()                


    def save_boxes(self):
        boxes = {f"box_{i}": {"xl": int(coords[0]), "yl": int(coords[1]), "xr": int(coords[2]), "yr": int(coords[3])} for i, coords in enumerate(self.rectangles)}
        # save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        save_path =  Path(self.current_path).parent
        # print(save_path)

        try:
            if save_path:
                with open(os.path.join(save_path, 'boxes.json') , 'w') as f:
                    json.dump(boxes, f)

            messagebox.showinfo("Information", "Boxes were saved in boxes.json")                
        
        except Exception as e:
            messagebox.showinfo(e)                

    def upload_boxes(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                boxes = json.load(f)
            self.clear_boxes()  # Clear existing rectangles
            for box in boxes.values():
                rect = [box["xl"], box["yl"], box["xr"], box["yr"]]
                self.rectangles.append(rect)
                # self.canvas.create_rectangle(*self.scale_coordinates(rect), outline='yellow')

        self.update_image()                



if __name__ == "__main__":
    root = tk.Tk()

    # img1 = 'T:\RetinalDatasets\TESTING\Eyemark\eyepacs_longitudinal_eyemark_output\20240118_eyemark_output\109\registered\images\0000109_t110_i006.jpg'
    # img2 = 'T:\RetinalDatasets\TESTING\Eyemark\eyepacs_longitudinal_eyemark_output\20240118_eyemark_output\109\registered\images\0000109_t110_i007.jpg'

    # img1 = '0000109_t110_i006.jpg'
    # img2 = '0000109_t110_i007.jpg'

    # img1 = '0000109_t110_i006.jpg'
    # img2 = '0000109_t110_i007.jpg'


    app = ImageToggleApp(root) #, img1, img2)
    root.mainloop()






# import tkinter as tk

# root = tk.Tk()
# root.title("Test Tkinter")
# root.geometry("200x100")
# label = tk.Label(root, text="Tkinter is working!")
# label.pack(pady=20)
# root.mainloop()