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
import yaml
import numpy as np

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

file_path = './mask_overlay_config.yaml'
config = read_yaml(file_path)

predicted_mask_name = config['predicted_mask']
gt_mask_name = config.get('gt_mask', None)
rect_fill_color = config.get('fillcolor', None)
which_set = config.get('which_set', 'test')


def overlay_mask(image, mask, gt_mask, mask_color_pred=(0, 255, 255, 128), mask_color_gt=(0, 128, 255, 255), overlap_color = (0, 0, 0, 255)):
    # def overlay_mask(image, mask, gt_mask, mask_color_pred=(0, 255, 0, 0), mask_color_gt=(0, 0, 0, 255), overlap_color = (0, 255, 0)):
    # Load the image and the mask
    # breakpoint()
    mask = mask.convert("L")  # Convert mask to grayscale ('L' mode)

    # Ensure mask is binary (0 or 255)
    # mask = mask.point(lambda p: p > 0 and 255) 
    mask = mask.point(lambda p: p > 0 and 255) 
    # I[I>0] = 255 
    # mask = mask.point(lambda p: p > 50 and 255) 

    mask = mask.resize((image.size[0], image.size[1]))#, Image.Resampling.LANCZOS)


    # Create colored versions of both masks
    mask_colored1 = Image.new("RGBA", image.size, mask_color_pred)
    mask_colored1.putalpha(mask)  # Apply the first mask to the first colored image

    final_image = Image.alpha_composite(image.convert("RGBA"), mask_colored1)    

    

    if gt_mask is not None:
        gt_mask = gt_mask.convert("L")  # Convert mask to grayscale ('L' mode)
        gt_mask = gt_mask.resize((image.size[0], image.size[1]), Image.Resampling.LANCZOS)    

        mask_colored2 = Image.new("RGBA", image.size, mask_color_gt)
        mask_colored2.putalpha(gt_mask)  # Apply the second mask to the second colored image

        # Create an overlap mask where both mask1 and mask2 have non-zero values
        # overlap_mask = Image.new("L", image.size)  # "L" mode is for grayscale (single channel)
        # overlap_mask_data = overlap_mask.load()

        # Mark where both masks overlap
        # breakpoint()
        mask =  np.array(mask)
        gt_mask =  np.array(gt_mask)

        # for x in range(image.size[0]):
        #     for y in range(image.size[1]):
        #         if mask[x, y] > 0 and gt_mask[x, y] > 0:
        #             overlap_mask_data[x, y] = 255  # Fully opaque where masks overlap

        overlap_mask_np = np.where((mask > 0) & (gt_mask > 0), 255, 0).astype(np.uint8)
        overlap_mask_L = Image.fromarray(overlap_mask_np, mode="L")
        overlap_mask_L = overlap_mask_L.resize((image.size[0], image.size[1]), Image.Resampling.LANCZOS)        


        # Create the colored overlap mask
        overlap_mask = Image.new("RGBA", image.size, overlap_color)
        overlap_mask.putalpha(overlap_mask_L)  # Apply the overlap mask to the colored image



        # Combine both colored masks
        combined_mask = Image.alpha_composite(mask_colored1, mask_colored2)

        # Now combine the overlap mask with the other masks
        final_combined = Image.alpha_composite(combined_mask, overlap_mask)    
        
        # blend_factor=.5
        # final_combined = Image.blend(combined_mask, overlap_mask, blend_factor)




        # Overlay the combined mask on the original image
        final_image = Image.alpha_composite(image.convert("RGBA"), final_combined)

    return final_image


class ImageToggleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Change Annotator")
        self.root.state('zoomed')
        # self.root.bind_all("<Tab>", lambda e: "break")


        self.current_index = 0
        self.subfolders = []

        self.folder_path = None
        self.pan_status = 'ended'     

        # Initialize images
        self.image_a = None
        self.image_b = None
        self.current_image = None
        self.change_mask = None        
        self.gt_mask = None                
        self.current_path = None

        self.image_path_1 = None
        self.image_path_2 = None


        # Initialize zoom factor
        self.zoom_factor = .9 #1.0
        self.scale_factor = 1.1
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self.hidden_state = True
        self.overlay_hidden_state = False
        
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

        # self.lc_button = tk.Button(self.button_frame, text="LC", command=self.go_last_completed, width=3) #, state=tk.ENABLED)
        # self.lc_button.pack(side=tk.LEFT, padx=2, pady=10)

        # Label to display the counter position
        self.counter_label = tk.Label(self.button_frame)
        self.counter_label.pack(side=tk.LEFT, padx=(25, 0))


        # Create the toggle button
        # self.toggle_button = tk.Button(self.button_frame, text="Toggle Image", command=self.toggle_image)
        # self.toggle_button.pack(side=tk.LEFT, padx=150, pady=10)


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


        # Label to display the mouse position
        self.mask_name_label = tk.Label(self.button_frame) #, fg = "#%02x%02x%02x" % (255, 255, 128))
        self.mask_name_label.pack(side=tk.LEFT, padx=(85, 20))
        self.mask_name_label.config(text=f"Mask Name: {predicted_mask_name}, GT Mask Name: {gt_mask_name}")    

        # Label to display the mouse position
        self.position_label = tk.Label(self.button_frame)
        self.position_label.pack(side=tk.RIGHT, padx=(85, 20))



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
        self.canvas.pack()


        # Bind key event for toggling image
        self.root.bind("<space>", self.toggle_image)
        self.root.bind("<Control-=>", self.zoom_in)
        self.root.bind("<Control-minus>", self.zoom_out)        
        self.root.bind("<Control-0>", self.zoom_reset)                

        self.root.bind("<m>", self.hide_overlays)                        

        self.root.bind("<Right>", self.go_next)                
        self.root.bind("<Left>", self.go_back)   
        self.root.bind("<Home>", self.go_first)                
        self.root.bind("<End>", self.go_last)   
        # self.root.bind("<l>", self.go_last_completed)           

        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.zoom)    
        self.canvas.bind("<Button-2>", self.toggle_image)    

        self.canvas.bind("<Motion>", self.show_mouse_position)

        # Initialize drawing state
        self.start_x = None
        self.start_y = None
        self.rect = None


    def show_mouse_position(self, event):
        x, y = event.x, event.y
        self.position_label.config(text=f"(x,y): {x}, {y}")


    def upload_images(self):
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


        # load the change mask
        change_mask_path = os.path.join(self.label.cget("text"), predicted_mask_name)
        gt_mask_path = os.path.join(self.label.cget("text"), gt_mask_name if gt_mask_name else '')
        if os.path.exists(change_mask_path):        
            self.change_mask = Image.open(change_mask_path)
            self.gt_mask = Image.open(gt_mask_path) if (os.path.exists(gt_mask_path) and gt_mask_path.endswith('.png')) else None
        else:
            self.change_mask, self.gt_mask = None, None

        self.update_image()
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


        if self.overlay_hidden_state and self.change_mask:
            self.current_image = overlay_mask(self.current_image, self.change_mask, self.gt_mask)

        self.update_image()


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


    def update_image(self):
        img_width, img_height = self.current_image.size
        new_width = int(img_width * self.zoom_factor)
        new_height = int(img_height * self.zoom_factor)

        self.photo = ImageTk.PhotoImage(self.current_image.resize((new_width, new_height)))
        self.image_on_canvas = self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor=tk.NW, image=self.photo)


    def hide_overlays(self, event=None):

        self.overlay_hidden_state = not self.overlay_hidden_state

        if self.overlay_hidden_state and self.change_mask:
            self.current_image = overlay_mask(self.current_image, self.change_mask, self.gt_mask)

        else:
            self.current_image = Image.open(self.current_path)

        self.update_image()


    def zoom_in(self, event=None):
        self.zoom_factor *= self.scale_factor
        self.center_image()


    def zoom_out(self, event=None):
        self.zoom_factor /= self.scale_factor
        self.center_image()


    def zoom_reset(self, event=None):
        self.zoom_factor = .9 #1.0
        self.center_image()


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

        self.pan_status = 'panning'


    def end_pan(self, event):
        self.pan_status = 'ended'


    def center_image(self):
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


    def upload_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            # self.subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir()]
            with open(os.path.join(self.folder_path, f'{which_set}.txt'), 'r') as f:
                self.subfolders = f.readlines()
            self.subfolders = [os.path.join(self.folder_path, x.strip()) for x in self.subfolders]

            self.counter_label.config(text=f"1 out of {len(self.subfolders)}")

            # create the index file in the current directory if it does not exist
            self.current_index = 0

            self.update_label()
            self.update_buttons()
            self.upload_images()


    def go_first(self, event=None):
        if self.current_index > 0:
            # self.save_boxes()
            self.current_index = 0
            self.update_label()
            self.update_buttons()
            # self.upload_images()
            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")
            self.overlay_hidden_state = False


    def go_last(self, event=None):
        if self.current_index < len(self.subfolders) - 1:
            # self.save_boxes()
            self.current_index = len(self.subfolders) - 1
            self.update_label()
            self.update_buttons()
            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")
            self.overlay_hidden_state = False


    def go_back(self, event=None):
        if self.current_index > 0:
            # self.save_boxes()
            self.current_index -= 1
            self.update_label()
            self.update_buttons()
            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")
            self.overlay_hidden_state = False


    def go_next(self, event=None):
        if self.current_index < len(self.subfolders) - 1:

            self.current_index += 1
            with open(os.path.join(self.folder_path,'last_index.txt'), 'w') as f: 
                f.write(str(self.current_index))

            self.update_label() 
            self.update_buttons()

            self.upload_images()

            self.counter_label.config(text=f"{self.current_index+1} out of {len(self.subfolders)}")
            self.overlay_hidden_state = False


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


    def show_shortcuts(self):
        shortcuts = [
            'home key: Move to the first image',
            'left arrow key: Move to the previous image',
            'right arrow key : Move to the next image',
            'end key: Move to the last image',
            'Ctrl + = : Zoom in',
            'Ctrl + - : Zoom out',
            'Ctrl + 0 : Reset zoom',
            'm : Hide and unhide masks',
            'Space bar : Toggle between two images',
            'Mouse-roller click : Toggle between two images',
            'Mouse-roller forward and back : Zoom in and out',
            'Left click and drag : Drag the image around',
        ]
        
        shortcuts_message = "\n".join(shortcuts)
        messagebox.showinfo("Shortcuts", shortcuts_message)


if __name__ == "__main__":
    root = tk.Tk()

    app = ImageToggleApp(root) #, img1, img2)
    root.mainloop()
