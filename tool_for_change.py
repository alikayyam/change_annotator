# Copyright Eyenuk LLC
# Author:  Borji
# Purpose: To characterize changes in a pair of retinal images
# Created: 07/10/2013

import tkinter as tk
from tkinter import Canvas, filedialog, messagebox, ttk, font
from PIL import Image, ImageTk
import json
from pathlib import Path
import os
import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

file_path = './config.yaml'
config = read_yaml(file_path)

pred_name = config['pred_name']
color_codes = config['colors']


def overlay_mask(image, mask, mask_color=(0, 255, 255, 128)):
    # Load the image and the mask
    image = image.convert("RGBA")  # Convert image to RGBA to support transparency
    mask = mask.convert("L")  # Convert mask to grayscale ('L' mode)

    # Ensure mask is binary (0 or 255)
    mask = mask.point(lambda p: p > 128 and 255) 

    # Create an RGBA image for the mask with the specified color and apply the mask
    # breakpoint()
    mask = mask.resize((image.size[0], image.size[1]))#, Image.Resampling.LANCZOS)

    mask_colored = Image.new("RGBA", image.size, mask_color)
    mask_colored.putalpha(mask)  # Apply the mask to the colored image

    # Overlay the mask on the image
    result = Image.alpha_composite(image, mask_colored)

    return result


class ImageToggleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lesion Labeling Tool")
        self.root.state('zoomed')

        self.current_index = 0
        # self.image_list = []
        self.subfolders = []

        self.folder_path = None
        self.pan_status = 'ended'     

        # Initialize images
        self.current_image = None
        self.boxes_path = None

        self.image_path_1 = None
        self.image_path_2 = None


        # Initialize zoom factor
        self.zoom_factor = .9 # 1.0
        self.scale_factor = 1.1
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self.hidden_state = True
        self.overlay_hidden_state = False


        self.comment = ''

        # Create a frame for the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Create a bold font
        self.bold_font = font.Font(family="Helvetica", size=12, weight="bold")

        # Create and pack the buttons
        self.upload_button = tk.Button(self.button_frame, text="Upload Images", command=self.upload_folder)
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


        # Create the clear button
        self.clear_button = tk.Button(self.button_frame, text="Delete Boxes", command=self.clear_boxes)
        self.clear_button.pack(side=tk.RIGHT, padx=35, pady=10)


        # self.color_codes = {'tp': 'blue', 'fp':'red', 'fn':'Orange', 'q':'black'}
        self.color_codes = color_codes

        self.radio_var = tk.StringVar(value="tp")  # Initial value
        self.radio_tp = tk.Radiobutton(self.button_frame, text="True Positive", variable=self.radio_var, value="tp", fg = self.color_codes['tp']) #font=self.bold_font,
        self.radio_fp = tk.Radiobutton(self.button_frame, text="False Positive", variable=self.radio_var, value="fp", fg = self.color_codes['fp'])
        self.radio_tn = tk.Radiobutton(self.button_frame, text="False Negative (Miss)", variable=self.radio_var, value="fn", fg = self.color_codes['fn'])
        self.radio_else = tk.Radiobutton(self.button_frame, text="Questionable", variable=self.radio_var, value="q", fg = self.color_codes['q'])

        self.radio_tp.pack(side=tk.LEFT,  padx=(220, 3))
        self.radio_fp.pack(side=tk.LEFT, padx=3, pady=10)
        self.radio_tn.pack(side=tk.LEFT, padx=3, pady=10)
        self.radio_else.pack(side=tk.LEFT, padx=3, pady=10)


        # for announcing whether there was a change or not in images
        self.mismatch_var = False #tk.BooleanVar()
        self.mismatchbox = tk.Checkbutton(self.button_frame, text="mismatch?", variable=self.mismatch_var, command=self.mismatch)
        self.mismatchbox.pack(side=tk.LEFT, padx=(185, 0))


        # Create a label to display the selected file name
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(side=tk.TOP, fill=tk.X)

        self.comment_button = tk.Button(self.label_frame, text="View/Add Comment", command=self.open_comment_window)
        self.comment_button.pack(side=tk.RIGHT, padx=(20, 5))


        # self.label_01 = tk.Label(self.label_frame, text="Image:", wraplength=500,  fg="black")
        # self.label_01.pack(side=tk.LEFT,padx=(5,2),pady=10)

        # self.label_1 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black", width = 50)
        # self.label_1.pack(side=tk.LEFT,padx=(5,5),pady=10)

        # Create and pack the label to show the current folder
        self.label0 = tk.Label(self.label_frame, text="Folder:", wraplength=100, fg="red")
        self.label0.pack(side=tk.LEFT,padx=(5,2),pady=10)        

        self.label1 = tk.Label(self.label_frame, text="No folder selected", wraplength=1000)
        self.label1.pack(side=tk.LEFT,padx=(0,5),pady=10)



        self.label_2 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black")
        self.label_2.pack(side=tk.RIGHT,padx=(0,2),pady=10)

        self.label_02 = tk.Label(self.label_frame, text="Image B:", wraplength=500,  fg="black")
        self.label_02.pack(side=tk.RIGHT,padx=(0,2),pady=10)

        self.label_1 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black")
        self.label_1.pack(side=tk.RIGHT,padx=(0,5),pady=10)

        self.label_01 = tk.Label(self.label_frame, text="Image A:", wraplength=500,  fg="black")
        self.label_01.pack(side=tk.RIGHT,padx=(0,2),pady=10)





        # Label to display the mouse position
        self.position_label = tk.Label(self.label_frame)
        self.position_label.pack(side=tk.RIGHT, padx=(0, 15))


        # Create a frame for the canvas
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create the canvas
        self.canvas = Canvas(self.canvas_frame, background="black", width = 3200, height = 2800)
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
        self.root.bind("<m>", self.hide_overlays)                        
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
        self.canvas.bind("<Button-2>", self.hide_overlays)    
        self.canvas.bind("<Control-Button-1>", self.delete_box)

        self.canvas.bind("<Motion>", self.show_mouse_position)

        # Initialize drawing state
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.rectangles = []
        self.rectangle_types = {}  # maps rectangle to its type e.g., (10,10,100,100):(tp)       

        self.clear_boxes()


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

        if self.overlay_hidden_state and self.mask:
            self.current_image = overlay_mask(self.current_image, self.mask)

        self.update_image()


        if not self.hidden_state:
            self.clear_canvas()
        else:
            self.update_rectangles()




    def open_comment_window(self):
        # Create a new window
        self.comment_window = tk.Toplevel(self.root)
        self.comment_window.title("Comment Window")
        
        # Add a text widget for the user to type the comment
        self.comment_text = tk.Text(self.comment_window, wrap='word', width=100, height=20)
        self.comment_text.insert('1.0', self.comment)

        self.comment_text.pack(padx=10, pady=10)
        
        # Add a submit button to close the comment window and print the comment
        self.submit_button = tk.Button(self.comment_window, text="Submit", command=self.submit_comment)
        self.submit_button.pack(pady=10)
        
        
    def submit_comment(self):
        # Get the comment from the text widget
        self.comment = self.comment_text.get("1.0", tk.END).strip()
        self.comment_window.destroy()


    def mismatch(self):
        self.mismatch_var = not self.mismatch_var



    def show_mouse_position(self, event):
        x, y = event.x, event.y
        self.position_label.config(text=f"(x,y): {x}, {y}") 


    def upload_images(self):
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        subfolder = self.subfolders[self.current_index].split('\\')[-1]
        # image_path = os.path.join(self.subfolders[self.current_index], subfolder + '.jpg')
        # self.current_image = Image.open(image_path)
        # self.current_image_size = self.current_image.size


        files = os.listdir(self.subfolders[self.current_index])
        imgs = [img for img in files if img.endswith('.jpg')]
        self.image_path_1 = os.path.join(self.subfolders[self.current_index], imgs[0])
        self.image_path_2 = os.path.join(self.subfolders[self.current_index], imgs[1])


        self.image_a = Image.open(self.image_path_1)
        self.image_b = Image.open(self.image_path_2)

        self.current_image = self.image_b
        self.current_path = self.image_path_2

        self.label_1.config(text=f"{Path(self.image_path_1).name}", fg="black")
        self.label_2.config(text=f"{Path(self.image_path_2).name}", fg="blue")


        mask_path = os.path.join(self.subfolders[self.current_index], pred_name)
        self.mask = Image.open(mask_path)

        # # read the masks
        # for le in lesions:
        #     image_path = os.path.join(self.subfolders[self.current_index], f'{le}_' + subfolder + '.png')
        #     self.all_images[le] = Image.open(image_path)


        self.boxes_path = os.path.join(self.subfolders[self.current_index], f'boxes_change.json')
        self.clear_boxes()  # Clear existing rectangles
        self.upload_boxes()
        
        # reset radio buttons
        self.radio_var.set('tp')

        self.update_image()
        self.zoom_reset()


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

        # show the boxes only when the hidden status is False
        for rect in self.rectangles:
            self.canvas.create_rectangle(*self.scale_coordinates(rect), outline=self.color_codes[self.rectangle_types[tuple(rect)]], tags='rects', width=1)

        
    def scale_coordinates(self, coords):
        return  [coord * self.zoom_factor + self.pan_offset_x if i % 2 == 0 else coord * self.zoom_factor + self.pan_offset_y for i, coord in enumerate(coords)]


    def start_draw(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline=self.color_codes[self.radio_var.get()], tags='rects', width=1)


    def draw(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)


    def end_draw(self, event):
        self.rectangles.append([int((t - self.pan_offset_x)/ self.zoom_factor) if i % 2 == 0 else int((t - self.pan_offset_y)/ self.zoom_factor) for i, t in enumerate(self.canvas.coords(self.rect))])
        self.rectangle_types[tuple(self.rectangles[-1])] = self.radio_var.get()
        self.rect = None


    def clear_boxes(self):
        self.clear_canvas()
        self.rectangles.clear()


    # def hide_overlays(self, event=None):
    #     if self.overlay_hidden_state:
    #         self.current_image = overlay_mask(self.current_image, self.mask)

    #     else:
    #         subfolder = self.subfolders[self.current_index].split('\\')[-1]
    #         # image_path = os.path.join(self.subfolders[self.current_index], subfolder + '.jpg')
    #         self.current_image = Image.open(image_path)

    #     self.update_image()
    #     if self.hidden_state:
    #         self.update_rectangles()

    #     self.overlay_hidden_state = not self.overlay_hidden_state


    def hide_overlays(self, event=None):

        self.overlay_hidden_state = not self.overlay_hidden_state

        if self.overlay_hidden_state and self.mask:
            self.current_image = overlay_mask(self.current_image, self.mask)

        else:
            self.current_image = Image.open(self.current_path)

        self.update_image()
        if self.hidden_state:
            self.update_rectangles()




    def hide_rects(self, event=None):
        # print(self.hidden_state)
        if self.hidden_state:
            # Hide all rectangles
            # breakpoint()
            # for rect in self.rectangles:
            #     print(rect)
            #     self.canvas.itemconfig(rect, state=tk.HIDDEN)

            self.save_boxes()
            self.clear_canvas()

        else:
            # for rect in self.rectangles:
            #     print(rect)
            #     self.canvas.itemconfig(rect, state=tk.NORMAL)            
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
        self.zoom_factor = .9
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
            data = {}
            boxes = {f"box_{i}": {"xl": int(coords[0]), "yl": int(coords[1]), "xr": int(coords[2]), "yr": int(coords[3]), "type": self.rectangle_types[tuple(coords)]} for i, coords in enumerate(self.rectangles) if coords}
            data['rectangles'] = boxes
            data['mismatch'] = self.mismatch_var
            data['comment'] = self.comment
            with open(self.boxes_path , 'w') as f:
                json.dump(data, f)

            # messagebox.showinfo("Information", "Boxes were saved in boxes.json")                
            
        except Exception as e:
            messagebox.showinfo('Error', e)                


    def upload_boxes(self):

        # self.boxes_path = os.path.join(self.label.cget("text"), 'boxes.json')
        if not os.path.exists(self.boxes_path):  # create a dummy json
            with open(self.boxes_path, 'w') as f:
                json.dump({'rectangles': {}, 'mismatch': False, 'comment': ''}, f)

        with open(self.boxes_path, 'r') as f:
            data = json.load(f)
            self.mismatch_var = data['mismatch']
            # print(self.mismatch_var)

            if self.mismatch_var: 
                self.mismatchbox.select()
                # return
            else:
                self.mismatchbox.deselect()

            boxes = data['rectangles']
            for box in boxes.values():
                rect = [box["xl"], box["yl"], box["xr"], box["yr"]]
                self.rectangles.append(rect)
                self.rectangle_types[tuple(rect)] = box['type']
            
            self.update_rectangles()

            self.comment = data['comment']

    def upload_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir()]

            self.label1.config(text=f"{self.subfolders[self.current_index]}", fg="black")


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
            # self.upload_images()
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
        pass 


    def update_buttons(self):
        self.back_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)
        self.first_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.last_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)


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
            'm : Hide and unhide masks',            
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
