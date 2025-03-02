# Copyright Eyenuk LLC
# Author:  Borji
# Purpose: To annotate changes in a pair of retinal images
# Created: 07/10/2013


import tkinter as tk
from tkinter import Canvas, filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
from pathlib import Path
import os
from tkinter import font


lesions = {'ex': 'exudates', 
           'cw': 'cottonwoolspots',
           'he': 'hemorrhages',
           'ma': 'microaneurysms'
           }



def overlay_mask(image, mask, mask_color=(0, 255, 255, 128)):
    # Load the image and the mask
    image = image.convert("RGBA")  # Convert image to RGBA to support transparency
    mask = mask.convert("L")  # Convert mask to grayscale ('L' mode)

    # Ensure mask is binary (0 or 255)
    mask = mask.point(lambda p: p > 128 and 255) 

    # Create an RGBA image for the mask with the specified color and apply the mask
    mask_colored = Image.new("RGBA", image.size, mask_color)
    mask_colored.putalpha(mask)  # Apply the mask to the colored image

    # Overlay the mask on the image
    result = Image.alpha_composite(image, mask_colored)

    # Save the result
    # result.save(output_path, "PNG")
    return result



class ImageToggleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lesion Labeling Tool")
        self.root.state('zoomed')

        self.current_index = 0
        # self.image_list = []
        self.subfolders = []
        self.all_images = {}

        self.folder_path = None
        self.pan_status = 'ended'     
        self.last_zoom_pos = None

        # Initialize images
        # self.image_a = None
        # self.image_b = None
        self.current_image = None
        # self.current_path = None

        # self.image_path_1 = None
        # self.image_path_2 = None

        self.boxes_path = None
        

        # Initialize zoom factor
        self.zoom_factor = 1.0
        self.scale_factor = 1.1
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self.hidden_state = True
        self.overlay_hidden_state = False

        # self.new_img = 0

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



        # Create the clear button
        self.clear_button = tk.Button(self.button_frame, text="Delete Boxes", command=self.clear_boxes)
        self.clear_button.pack(side=tk.LEFT, padx=35, pady=10)




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



        self.radio_lesion = tk.StringVar(value="ex")  # Initial value
        self.radio_ex = tk.Radiobutton(self.button_frame, text="Exudates", variable=self.radio_lesion, value="ex", command=self.radio_changed) #, font=self.bold_font)
        self.radio_he = tk.Radiobutton(self.button_frame, text="Hemoraghes", variable=self.radio_lesion, value="he", command=self.radio_changed) #, font=self.bold_font)
        self.radio_cw = tk.Radiobutton(self.button_frame, text="Cottonwool spots", variable=self.radio_lesion, value="cw", command=self.radio_changed) #, font=self.bold_font)
        self.radio_ma = tk.Radiobutton(self.button_frame, text="Microaneurysm", variable=self.radio_lesion, value="ma", command=self.radio_changed) #, font=self.bold_font)

        self.radio_ex.pack(side=tk.LEFT,  padx=(238, 5))
        self.radio_he.pack(side=tk.LEFT, padx=5, pady=10)
        self.radio_cw.pack(side=tk.LEFT, padx=5, pady=10)
        self.radio_ma.pack(side=tk.LEFT, padx=5, pady=10)



        # self.combo_label = tk.Label(self.button_frame, text="Lesion Type:", wraplength=500,  fg="black")
        # self.combo_label.pack(side=tk.LEFT,padx=(5,2),pady=10)
        # options = ["Option 1", "Option 2", "Option 3", "Option 4"]
        # self.combo_lesion = ttk.Combobox(self.button_frame, values=options)
        # self.combo_lesion.pack(side=tk.LEFT, padx=10, pady=10)





        # for announcing whether there was a change or not in images
        self.mismatch_var = False #tk.BooleanVar()
        self.mismatchbox = tk.Checkbutton(self.button_frame, text="mismatch?", variable=self.mismatch_var, command=self.mismatch)
        self.mismatchbox.pack(side=tk.LEFT, padx=(185, 0))







        # Create a label to display the selected file name
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(side=tk.TOP, fill=tk.X)

        # # Create and pack the label to show the current folder
        # self.label0 = tk.Label(self.label_frame, text="Folder:", wraplength=100, fg="red")
        # self.label0.pack(side=tk.LEFT,padx=(5,2),pady=10)        

        # self.label = tk.Label(self.label_frame, text="No folder selected", wraplength=1000)
        # self.label.pack(side=tk.LEFT,padx=(0,5),pady=10)

        self.label_01 = tk.Label(self.label_frame, text="Image:", wraplength=500,  fg="black")
        self.label_01.pack(side=tk.LEFT,padx=(5,2),pady=10)

        self.label_1 = tk.Label(self.label_frame, text="No image selected", wraplength=500,  fg="black", width = 50)
        self.label_1.pack(side=tk.LEFT,padx=(5,5),pady=10)





        self.color_codes = {'tp': 'blue', 'fp':'red', 'fn':'Orange', 'q':'black'}

        self.radio_var = tk.StringVar(value="tp")  # Initial value
        self.radio_tp = tk.Radiobutton(self.label_frame, text="True Positive", variable=self.radio_var, value="tp", fg = self.color_codes['tp']) #font=self.bold_font,
        self.radio_fp = tk.Radiobutton(self.label_frame, text="False Positive", variable=self.radio_var, value="fp", fg = self.color_codes['fp'])
        self.radio_tn = tk.Radiobutton(self.label_frame, text="False Negative", variable=self.radio_var, value="fn", fg = self.color_codes['fn'])
        self.radio_else = tk.Radiobutton(self.label_frame, text="Questionable", variable=self.radio_var, value="q", fg = self.color_codes['q'])

        self.radio_tp.pack(side=tk.LEFT,  padx=(261, 5))
        self.radio_fp.pack(side=tk.LEFT, padx=5, pady=10)
        self.radio_tn.pack(side=tk.LEFT, padx=5, pady=10)
        self.radio_else.pack(side=tk.LEFT, padx=5, pady=10)







        # Label to display the mouse position
        self.position_label = tk.Label(self.label_frame)
        self.position_label.pack(side=tk.RIGHT, padx=(0, 15))




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
        # self.root.bind("<space>", self.toggle_image)
        self.root.bind("<Control-=>", self.zoom_in)
        self.root.bind("<Control-minus>", self.zoom_out)        
        self.root.bind("<Control-0>", self.zoom_reset)                

        self.root.bind("<Control-h>", self.hide_rects)                
        self.root.bind("<space>", self.hide_overlays)                        
        # self.root.bind("<Shift-U>", self.unhide_rects)                        

        self.root.bind("<Control-Right>", self.go_next)                
        self.root.bind("<Control-Left>", self.go_back)   
        self.root.bind("<Control-Home>", self.go_first)                
        self.root.bind("<Control-End>", self.go_last)   
        self.root.bind("<Control-l>", self.go_last_completed)           

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
        self.rectangle_types = {}  # maps rectangle to its type e.g., (10,10,100,100):(ex,tp)       

        self.clear_boxes()


    def mismatch(self):
        self.mismatch_var = not self.mismatch_var


    def radio_changed(self):
        # selected_value = self.radio_var.get()
        # self.value_label.config(text=f"Selected Value: {selected_value}")
        self.update_rectangles()
        self.overlay_hidden_state = False
        self.hide_overlays()


    def show_mouse_position(self, event):
        x, y = event.x, event.y
        self.position_label.config(text=f"(x,y): {x}, {y}")


    def upload_images(self):
        # self.subfolder_path = os.path.join(self.folder_path, self.subfolders[self.current_index]) # image name should be the same as the subfolder name and jpg

        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        # breakpoint()

        subfolder = self.subfolders[self.current_index].split('\\')[-1]
        image_path = os.path.join(self.subfolders[self.current_index], subfolder + '.jpg')
        self.current_image = Image.open(image_path)

        # read the masks
        for u,v in lesions.items():
            image_path = os.path.join(self.subfolders[self.current_index], f'{v}_' + subfolder + '.png')
            self.all_images[u] = Image.open(image_path)


        self.label_1.config(text=f"{self.subfolders[self.current_index]}", fg="black")

        subfolder = self.subfolders[self.current_index].split('\\')[-1]
        self.boxes_path = os.path.join(self.subfolders[self.current_index], f'{subfolder}_boxes.json')
        self.clear_boxes()  # Clear existing rectangles
        self.upload_boxes()

        
        # reset radio buttons
        self.radio_lesion.set('ex')
        self.radio_var.set('tp')

        self.update_image()
        self.zoom_reset()



    def clear_canvas(self):
        self.canvas.delete("rects")  # Delete all items on the canvas            


    def zoom(self, event):
        
        # Get the current image size
        img_width, img_height = self.current_image.size

        x, y = event.x, event.y
        if (x, y) != self.last_zoom_pos: 
            print(x, y, self.last_zoom_pos)

            self.last_zoom_pos = (x, y)
            return
        self.last_zoom_pos = (x, y)


        # print(x, y, self.pan_offset_x, self.pan_offset_y)
        # Calculate the zoom direction and factor
        if event.delta < 0:  # Zoom in
            self.zoom_factor /= self.scale_factor
        else:  # Zoom out
            self.zoom_factor *= self.scale_factor

        # Calculate new size
        new_width = int(img_width * self.zoom_factor)
        new_height = int(img_height * self.zoom_factor)


        # Resize image
        tmp_image = self.current_image.resize((new_width, new_height))
        self.photo = ImageTk.PhotoImage(tmp_image) #, Image.Resampling.LANCZOS))
        
        # print(x,y)

        self.pan_offset_x = x - x * self.zoom_factor
        self.pan_offset_y = y - y * self.zoom_factor

        # print(new_x, new_y)
        # Update image on canvas
        self.canvas.delete(self.image_on_canvas)
        self.image_on_canvas = self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor="nw", image=self.photo)


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
            if self.rectangle_types[tuple(rect)][0] == self.radio_lesion.get():
                self.canvas.create_rectangle(*self.scale_coordinates(rect), outline=self.color_codes[self.rectangle_types[tuple(rect)][1]], tags='rects', width=1)


    def scale_coordinates(self, coords):
        # return  [coord * self.zoom_factor + self.pan_offset_x if i % 2 == 0 else coord * self.zoom_factor + self.pan_offset_y for i, coord in enumerate(coords)]
        return  [coord * self.zoom_factor + self.pan_offset_x if i % 2 == 0 else coord * self.zoom_factor + self.pan_offset_y for i, coord in enumerate(coords)]    


    def start_draw(self, event):
        self.start_x = event.x
        self.start_y = event.y
        print(self.radio_var)
        # breakpoint()
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline=self.color_codes[self.radio_var.get()], tags='rects', width=1)


    def draw(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)


    def end_draw(self, event):
        self.rectangles.append([(t - self.pan_offset_x)/ self.zoom_factor if i % 2 == 0 else (t - self.pan_offset_y)/ self.zoom_factor for i, t in enumerate(self.canvas.coords(self.rect))])
        self.rectangle_types[tuple(self.rectangles[-1])] = (self.radio_lesion.get(), self.radio_var.get())
        self.rect = None


    def clear_boxes(self):
        self.clear_canvas()
        self.rectangles.clear()



    def hide_overlays(self, event=None):
        if self.overlay_hidden_state:
            mask = self.all_images[self.radio_lesion.get()]
            self.current_image = overlay_mask(self.current_image, mask)

        else:
            subfolder = self.subfolders[self.current_index].split('\\')[-1]
            image_path = os.path.join(self.subfolders[self.current_index], subfolder + '.jpg')
            self.current_image = Image.open(image_path)

        self.update_image()
        if self.hidden_state:
            self.update_rectangles()


        self.overlay_hidden_state = not self.overlay_hidden_state



    def hide_rects(self, event=None):
        print(self.hidden_state)
        if self.hidden_state:
            # Hide all rectangles
            for rect in self.rectangles:
                self.canvas.itemconfig(rect, state=tk.HIDDEN)
            self.clear_canvas()

        else:
            for rect in self.rectangles:
                self.canvas.itemconfig(rect, state=tk.NORMAL)            
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
        self.zoom_factor = 1.0
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
        print('start pan')


    def pan(self, event):
        print(event.x, event.y, self.pan_offset_x, self.pan_offset_y)
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

        # self.update_image()
        self.pan_status = 'panning'
        print('panning')        

    def end_pan(self, event):
        self.pan_status = 'ended'
        print('pan ended')       
        print(self.pan_offset_x, self.pan_offset_y) 


    def center_image_2(self):
        print(self.zoom_factor)

        # Get the current image size
        img_width, img_height = self.current_image.size

        new_width = int(self.current_image.width * self.zoom_factor)
        new_height = int(self.current_image.height * self.zoom_factor)

        self.pan_offset_x -= new_width - img_width
        self.pan_offset_y -= new_height - img_height

        tmp_image = self.current_image.resize((new_width, new_height))
        self.photo = ImageTk.PhotoImage(tmp_image) #, Image.Resampling.LANCZOS))
        self.canvas.delete(self.image_on_canvas)
        self.image_on_canvas = self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor="nw", image=self.photo)



    def center_image(self):
        print(self.zoom_factor)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = int(self.current_image.width * self.zoom_factor)
        image_height = int(self.current_image.height * self.zoom_factor)
        self.pan_offset_x = (canvas_width - image_width) / 2 
        self.pan_offset_y = (canvas_height - image_height) / 2 
        # self.canvas.coords(self.image_on_canvas, self.pan_offset_x, self.pan_offset_y)

        tmp_image = self.current_image.resize((image_width, image_height))
        self.photo = ImageTk.PhotoImage(tmp_image) #, Image.Resampling.LANCZOS))
        self.canvas.delete(self.image_on_canvas)
        self.image_on_canvas = self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor="nw", image=self.photo)

        # self.canvas.coords(self.image_on_canvas, (canvas_width - image_width) // 2, (canvas_height - image_width) // 2)


    def delete_box(self, event):
        # self.delete_box_mode = True
        for i, rect in enumerate(self.rectangles):
            x1, y1, x2, y2 = self.scale_coordinates(rect)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                # self.canvas.delete(self.canvas.find_closest((x1 + x2) / 2, (y1 + y2) / 2))
                self.rectangles.remove(rect)
                # print(self.pan_offset_x, self.pan_offset_y)
                print(len(self.rectangles))
                break
        
        self.update_rectangles()
        # self.update_image()      
        # self.delete_box_mode = False


    def save_boxes(self):
        try:
            # if self.boxes_path:
            data = {}
            boxes = {f"box_{i}": {"xl": int(coords[0]), "yl": int(coords[1]), "xr": int(coords[2]), "yr": int(coords[3]), "types": self.rectangle_types[tuple(coords)]} for i, coords in enumerate(self.rectangles) if coords}
            data['rectangles'] = boxes
            data['mismatch'] = self.mismatch_var
            with open(self.boxes_path , 'w') as f:
                json.dump(data, f)

            # messagebox.showinfo("Information", "Boxes were saved in boxes.json")                
            
        except Exception as e:
            messagebox.showinfo('Error', e)                


    def upload_boxes(self):

        # self.boxes_path = os.path.join(self.label.cget("text"), 'boxes.json')
        if not os.path.exists(self.boxes_path):  # create a dummy json
            with open(self.boxes_path, 'w') as f:
                json.dump({'rectangles': {}, 'mismatch': False}, f)

        with open(self.boxes_path, 'r') as f:
            data = json.load(f)
            self.mismatch_var = data['mismatch']
            print(self.mismatch_var)

            if self.mismatch_var: 
                self.mismatchbox.select()
                # return
            else:
                self.mismatchbox.deselect()

            boxes = data['rectangles']
            for box in boxes.values():
                rect = [box["xl"], box["yl"], box["xr"], box["yr"]]
                self.rectangles.append(rect)
                self.rectangle_types[tuple(rect)] = box['types']
            
            self.update_rectangles()


        # with open(self.boxes_path, 'r') as f:
        #     boxes = json.load(f)
        # self.clear_boxes()  # Clear existing rectangles
        # for box in boxes.values():
        #     rect = [box["xl"], box["yl"], box["xr"], box["yr"]]
        #     self.rectangles.append(rect)
        #     self.rectangle_types[tuple(rect)] = box['types']
        #     # self.canvas.create_rectangle(*self.scale_coordinates(rect), outline='yellow')

        # self.update_rectangles()
        # # self.update_image()                


    def upload_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir()]
            # self.image_list = [im for im in os.listdir(self.folder_path) if im.endswith('.jpg')]
            # breakpoint()
            # print(self.image_list)
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



    def go_last(self, event=None):
        if self.current_index < len(self.subfolders) - 1:
            self.save_boxes()
            self.current_index = len(self.subfolders) - 1
            self.update_label()
            self.update_buttons()
            # self.upload_images()
            self.upload_images()


    def go_last_completed(self, event=None):
        self.save_boxes()

        with open(os.path.join(self.folder_path,'last_index.txt'), 'r') as f: 
            self.current_index = int(f.read())         

        self.update_label()
        self.update_buttons()
        self.upload_images()





    def go_back(self, event=None):
        if self.current_index > 0:
            self.save_boxes()
            self.current_index -= 1
            self.update_label()
            self.update_buttons()
            self.upload_images()



    def go_next(self, event=None):
        if self.current_index < len(self.subfolders) - 1:
            self.save_boxes()

            self.current_index += 1
            with open(os.path.join(self.folder_path,'last_index.txt'), 'w') as f: 
                f.write(str(self.current_index))

            self.update_label() 
            self.update_buttons()

            self.upload_images()


    def update_label(self):
        pass 
        # if self.subfolders:
        #     current_folder = self.subfolders[self.current_index]
        #     self.label.config(text=current_folder)
        # else:
        #     self.label.config(text="No subfolders found")


    def update_buttons(self):
        self.back_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)
        self.first_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.last_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)








    def show_shortcuts(self):
        shortcuts = [
            'Ctrl + up arrow key: Move to the first image',
            'Ctrl + left arrow key: Move to the previous image',
            'Ctrl + right arrow key : Move to the next image',
            'Ctrl + down arrow key: Move to the last image',
            'Ctrl + l: Move to the last completed (LC) image',
            'Ctrl + = : Zoom in',
            'Ctrl + - : Zoom out',
            'Ctrl + 0 : Reset zoom',
            'Ctrl + h : Hide and unhide boxes',
            'Space bar : Hide and unhide lesions',
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
