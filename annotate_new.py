# Copyright Eyenuk LLC
# Author:  Borji
# Purpose: To annotate changes in a pair of retinal images
# Created: 07/10/2013


import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import json
from pathlib import Path
import os


which_set = 'test_selected'

class ImageToggleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Change Annotator")
        self.root.state('zoomed')

        self.current_index = 0
        self.subfolders = []

        self.folder_path = None
        self.pan_status = 'ended'     

        # Initialize images
        self.image_a = None
        self.image_b = None
        self.current_image = None
        self.current_path = None

        self.image_path_1 = None
        self.image_path_2 = None

        self.boxes_path = None
        
        self.box_size = 10
        self.delete_box_size = False

        # Initialize zoom factor
        self.zoom_factor = .9 #1.0
        self.scale_factor = 1.1
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self.hidden_state = True

        # Create a frame for the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Create and pack the buttons
        self.upload_button = tk.Button(self.button_frame, text="Upload Root Folder", command=self.upload_folder)
        self.upload_button.pack(side=tk.LEFT, padx=(5,20), pady=10)

        self.first_button = tk.Button(self.button_frame, text="|<", command=self.go_first, state=tk.DISABLED, width=3)
        self.first_button.pack(side=tk.LEFT, padx=2, pady=10)

        self.back_button = tk.Button(self.button_frame, text="<", command=self.go_back, state=tk.DISABLED, width=3)
        self.back_button.pack(side=tk.LEFT, padx=2, pady=10)

        self.next_button = tk.Button(self.button_frame, text=">", command=self.go_next, state=tk.DISABLED, width=3)
        self.next_button.pack(side=tk.LEFT, padx=2, pady=10)

        self.last_button = tk.Button(self.button_frame, text=">|", command=self.go_last, state=tk.DISABLED, width=3)
        self.last_button.pack(side=tk.LEFT, padx=2, pady=10)

        self.lc_button = tk.Button(self.button_frame, text="LC", command=self.go_last_completed, width=3) #, state=tk.ENABLED)
        self.lc_button.pack(side=tk.LEFT, padx=2, pady=10)


        # Label to display the counter position
        self.counter_label = tk.Label(self.button_frame)
        self.counter_label.pack(side=tk.LEFT, padx=(25, 0))


        # Create the toggle button
        self.toggle_button = tk.Button(self.button_frame, text="Toggle Image", command=self.toggle_image)
        self.toggle_button.pack(side=tk.LEFT, padx=150, pady=10)

        # Create the clear button
        self.clear_button = tk.Button(self.button_frame, text="Delete Boxes", command=self.clear_boxes)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.shortcut_button = tk.Button(self.button_frame, text="Shortcuts", command=self.show_shortcuts)
        self.shortcut_button.pack(side=tk.RIGHT, padx=(30,5))



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
        self.upload_boxes_button = tk.Button(self.button_frame, text="Load Boxes", command=self.upload_boxes)
        self.upload_boxes_button.pack(side=tk.LEFT, padx=5, pady=10)


        self.box_label = tk.Label(self.button_frame, text="Box size:")
        self.box_label.pack(side=tk.LEFT, padx=(50,5), pady=10)
        self.entry = tk.Entry(self.button_frame,  width=5)
        self.entry.pack(side=tk.LEFT, padx=5,pady=10)
        self.entry.insert(0, '10')  # Set initial value to 100
        # Bind the Enter key to the update_global_number function
        self.entry.bind('<Return>', self.update_global_number)        


        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(self.button_frame, text="Delete", variable=self.checkbox_var, command=self.checkbox_changed)
        self.checkbox.pack(side=tk.LEFT, padx=(5, 0))


        # for announcing whether there was a change or not in images
        self.change_var = False
        self.changebox = tk.Checkbutton(self.button_frame, text="Discard", variable=self.change_var, command=self.changebox_changed)
        self.changebox.pack(side=tk.LEFT, padx=(35, 0))


        # Label to display the mouse position
        self.position_label = tk.Label(self.button_frame)
        self.position_label.pack(side=tk.LEFT, padx=(85, 0))


        # Create a label to display the selected file name
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(side=tk.TOP, fill=tk.X)

        # Create and pack the label to show the current folder
        self.label0 = tk.Label(self.label_frame, text="Folder:", wraplength=100, fg="red")
        self.label0.pack(side=tk.LEFT,padx=(5,2),pady=10)        

        self.label = tk.Label(self.label_frame, text="No folder selected", wraplength=1000)
        self.label.pack(side=tk.LEFT,padx=(0,5),pady=10)

        self.label_2 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black")
        self.label_2.pack(side=tk.RIGHT,padx=(0,2),pady=10)

        self.label_02 = tk.Label(self.label_frame, text="Image B:", wraplength=500,  fg="black")
        self.label_02.pack(side=tk.RIGHT,padx=(0,2),pady=10)

        self.label_1 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black")
        self.label_1.pack(side=tk.RIGHT,padx=(0,5),pady=10)

        self.label_01 = tk.Label(self.label_frame, text="Image A:", wraplength=500,  fg="black")
        self.label_01.pack(side=tk.RIGHT,padx=(0,2),pady=10)


        # Create a frame for the canvas
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create the canvas
        self.canvas = Canvas(self.canvas_frame, background="black", width = 3200, height = 2800)

        # self.canvas = tk.Canvas(self.canvas_frame, background="black", width = width,height = height)
        self.canvas.pack()

        # Bind mouse events for drawing bounding boxes
        self.canvas.bind("<Button-3>", self.start_draw)
        self.canvas.bind("<B3-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-3>", self.end_draw)

        # Bind key event for toggling image
        self.root.bind("<space>", self.toggle_image)
        self.root.bind("<Control-=>", self.zoom_in)
        self.root.bind("<Control-minus>", self.zoom_out)        
        self.root.bind("<Control-0>", self.zoom_reset)                

        self.root.bind("<h>", self.hide_rects)                
        # self.root.bind("<Shift-U>", self.unhide_rects)                        

        self.root.bind("<Right>", self.go_next)                
        self.root.bind("<Left>", self.go_back)   
        self.root.bind("<Home>", self.go_first)                
        self.root.bind("<End>", self.go_last)   
        self.root.bind("<l>", self.go_last_completed)           

        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.zoom)    
        self.canvas.bind("<Button-2>", self.toggle_image)    
        self.canvas.bind("<Control-Button-1>", self.delete_box)

        self.canvas.bind("<Motion>", self.show_mouse_position)


        # Initialize drawing state
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.rectangles = []

        self.clear_boxes()


    def show_mouse_position(self, event):
        x, y = event.x, event.y
        self.position_label.config(text=f"(x,y): {x}, {y}")


    def upload_images(self):
        # print(self.label.cget("text"))
        files = os.listdir(self.label.cget("text"))
        imgs = [img for img in files if img.endswith('.jpg')]
        self.image_path_1 = os.path.join(self.label.cget("text"), imgs[0])
        self.image_path_2 = os.path.join(self.label.cget("text"), imgs[1])

        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self.image_a = Image.open(self.image_path_1)
        self.image_b = Image.open(self.image_path_2)

        self.current_image = self.image_b
        self.current_path = self.image_path_2

        self.label_1.config(text=f"{Path(self.image_path_1).name}", fg="black")
        self.label_2.config(text=f"{Path(self.image_path_2).name}", fg="blue")


        self.clear_boxes()  # Clear existing rectangles
        self.upload_boxes()
        self.update_image()
        self.update_rectangles()
        self.zoom_reset()


    def toggle_image(self, event=None):
        if self.current_path == self.image_path_1:
            self.current_image = self.image_b
            self.current_path = self.image_path_2
            self.label_2.config(fg="blue")
            self.label_1.config(fg="black")
        else:
            self.current_image = self.image_a
            self.current_path = self.image_path_1
            self.label_1.config(fg="blue")
            self.label_2.config(fg="black")

        self.update_image()

        if not self.hidden_state:
            self.clear_canvas()
        else:
            self.update_rectangles()


    def clear_canvas(self):
        self.canvas.delete("rects")  # Delete all items on the canvas            


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

        if not self.hidden_state:            
            self.clear_canvas()
        else:
            self.update_rectangles()


    def update_image(self):
        img_width, img_height = self.current_image.size
        new_width = int(img_width * self.zoom_factor)
        new_height = int(img_height * self.zoom_factor)

        self.photo = ImageTk.PhotoImage(self.current_image.resize((new_width, new_height)))
        self.image_on_canvas = self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor=tk.NW, image=self.photo)


    def update_rectangles(self):
        # Clear canvas and redraw all rectangles on the new image
        self.clear_canvas()

        # if not self.hidden_state: return

        self.delete_box_size = self.checkbox_var.get()
        if self.delete_box_size:
            self.rectangles = [rect for rect in self.rectangles if (rect[2] - rect[0]) * (rect[3] - rect[1]) >= self.box_size]
        
        # show the boxes only when the hidden status is False
        for rect in self.rectangles:
            try:
                self.canvas.create_rectangle(*self.scale_coordinates(rect), outline='yellow', tags='rects')

            except Exception as e:
                messagebox.showinfo('Error', e)                 


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
        if self.hidden_state:
            self.save_boxes()
            self.clear_canvas()

        else:
            self.update_image()
            self.update_rectangles()

        self.hidden_state = not self.hidden_state


    def zoom_in(self, event=None):
        self.zoom_factor *= self.scale_factor
        self.center_image()
        if not self.hidden_state:            
            self.clear_canvas()
        else:
            self.update_rectangles()


    def zoom_out(self, event=None):
        self.zoom_factor /= self.scale_factor
        self.center_image()
        if not self.hidden_state:            
            self.clear_canvas()
        else:
            self.update_rectangles()


    def zoom_reset(self, event=None):
        self.zoom_factor = .9 #1.0
        self.center_image()
        if not self.hidden_state:            
            self.clear_canvas()
        else:
            self.update_rectangles()


    def start_pan(self, event):
        if self.pan_status != 'ended': 
            self.pan_status = 'ended'
            return

        self.mouse_x = event.x
        self.mouse_y = event.y

        self.pan_status = 'started'

    def pan(self, event):
        if self.pan_status not in ['started', 'panning']: 
            return

        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y

        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.canvas.move(self.image_on_canvas, dx, dy)

        self.mouse_x = event.x
        self.mouse_y = event.y

        self.update_rectangles()
        self.pan_status = 'panning'

    def end_pan(self, event):
        self.pan_status = 'ended'


    def center_image(self):
        # print(self.zoom_factor)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = int(self.current_image.width * self.zoom_factor)
        image_height = int(self.current_image.height * self.zoom_factor)
        self.pan_offset_x = (canvas_width - image_width) / 2 
        self.pan_offset_y = (canvas_height - image_height) / 2 

        tmp_image = self.current_image.resize((image_width, image_height))
        self.photo = ImageTk.PhotoImage(tmp_image) #, Image.Resampling.LANCZOS))
        self.canvas.delete(self.image_on_canvas)
        self.image_on_canvas = self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor="nw", image=self.photo)


    def delete_box(self, event):
        for i, rect in enumerate(self.rectangles):
            x1, y1, x2, y2 = self.scale_coordinates(rect)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.rectangles.remove(rect)
                break
        
        self.update_rectangles()


    def save_boxes(self):
        try:
            # if self.boxes_path:
            data = {}
            boxes = {f"box_{i}": {"xl": int(coords[0]), "yl": int(coords[1]), "xr": int(coords[2]), "yr": int(coords[3])} for i, coords in enumerate(self.rectangles) if coords}
            data['rectangles'] = boxes
            data['discard'] = self.change_var
            with open(self.boxes_path , 'w') as f:
                json.dump(data, f)

            # messagebox.showinfo("Information", "Boxes were saved in boxes.json")                
            
        except Exception as e:
            messagebox.showinfo('Error', e)                


    def upload_boxes(self):
        self.boxes_path = os.path.join(self.label.cget("text"), 'boxes_change.json')
        if not os.path.exists(self.boxes_path):  # create a dummy json
            with open(self.boxes_path, 'w') as f:
                json.dump({'rectangles': {}, 'discard': False}, f)

        with open(self.boxes_path, 'r') as f:
            data = json.load(f)
            self.change_var = data.get('discard', False)

            if self.change_var: 
                self.changebox.select()
            else: 
                self.changebox.deselect()

            
            boxes = data.get('rectangles', data) # to comply with the old format
            for box in boxes.values():
                rect = [box["xl"], box["yl"], box["xr"], box["yr"]]
                self.rectangles.append(rect)
            
        self.update_rectangles()


    def upload_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir()]

            # with open(os.path.join(self.folder_path, f'{which_set}.txt'), 'r') as f:
            #     self.subfolders = f.readlines()
            # self.subfolders = [os.path.join(self.folder_path, x.strip()) for x in self.subfolders]


            self.counter_label.config(text=f"1 out of {len(self.subfolders)}")

            # create the index file in the current directory if it does not exist
            if not os.path.exists(os.path.join(self.folder_path, 'last_index.txt')):
                with open(os.path.join(self.folder_path,'last_index.txt'), 'w') as f: 
                    f.write('0')
                self.current_index = 0
            else: # if it exists # read the last index file
                with open(os.path.join(self.folder_path,'last_index.txt'), 'r') as f: 
                    self.current_index = int(f.read())         

            self.update_label()
            self.update_buttons()
            self.upload_images()


    def go_first(self, event=None):
        if self.current_index > 0:
            self.save_boxes()
            self.current_index = 0
            self.update_label()
            self.update_buttons()
            # self.upload_images()
            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")


    def go_last(self, event=None):
        if self.current_index < len(self.subfolders) - 1:
            self.save_boxes()
            self.current_index = len(self.subfolders) - 1
            self.update_label()
            self.update_buttons()
            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")


    def go_last_completed(self, event=None):
        self.save_boxes()

        with open(os.path.join(self.folder_path,'last_index.txt'), 'r') as f: 
            self.current_index = int(f.read())         

        self.update_label()
        self.update_buttons()
        self.upload_images()

        self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")


    def go_back(self, event=None):
        if self.current_index > 0:
            self.save_boxes()
            self.current_index -= 1
            self.update_label()
            self.update_buttons()
            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")


    def go_next(self, event=None):
        if self.current_index < len(self.subfolders) - 1:
            self.save_boxes()

            self.current_index += 1
            with open(os.path.join(self.folder_path,'last_index.txt'), 'w') as f: 
                f.write(str(self.current_index))

            self.update_label() 
            self.update_buttons()

            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")



    def update_label(self):
        if self.subfolders:
            current_folder = self.subfolders[self.current_index]
            self.label.config(text=current_folder)
        else:
            self.label.config(text="No subfolders found")


    def update_buttons(self):
        self.back_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)
        self.first_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.last_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)


    def update_global_number(self, event):
        try:
            # Get the content from the Entry widget and convert it to a float
            self.box_size = float(self.entry.get())
            # messagebox.showinfo("Global Number Updated", f"Global number updated to: {global_number}")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")

        self.update_rectangles()
        self.canvas.focus_set()


    def checkbox_changed(self):
        # self.update_image()
        self.update_rectangles()
        self.canvas.focus_set()


    def changebox_changed(self):
        self.change_var = not self.change_var


    def show_shortcuts(self):
        shortcuts = [
            'home key: Move to the first image',
            'left arrow key: Move to the previous image',
            'right arrow key : Move to the next image',
            'end key: Move to the last image',
            'l: Move to the last completed (LC) image',
            'Ctrl + = : Zoom in',
            'Ctrl + - : Zoom out',
            'Ctrl + 0 : Reset zoom',
            'h : Hide and unhide boxes',
            'Space bar : Toggle between two images',
            'Mouse-roller click : Toggle between two images',
            'Mouse-roller forward and back : Zoom in and out',
            'Left click and drag : Drag the image around',
            'Ctrl + left click : Remove the box at the mouse location',
            'Right click and drag : Draw a box'
        ]
        
        shortcuts_message = "\n".join(shortcuts)
        messagebox.showinfo("Shortcuts", shortcuts_message)


if __name__ == "__main__":
    root = tk.Tk()

    app = ImageToggleApp(root) #, img1, img2)
    root.mainloop()
