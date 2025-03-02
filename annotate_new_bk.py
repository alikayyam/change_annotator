
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

        self.current_index = 0
        self.subfolders = []

        self.folder_path = None

        # Initialize images
        self.image_a = None
        self.image_b = None
        self.current_image = None
        self.current_path = None

        self.image_path_1 = None
        self.image_path_2 = None

        self.boxes_path = None
        self.change_path = None
        
        self.box_size = 10
        self.delete_box_size = False

        # Initialize zoom factor
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self.hidden_state = True

        self.new_img = 0

        # Create a frame for the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)



        # Create and pack the buttons
        self.upload_button = tk.Button(self.button_frame, text="Upload Root Folder", command=self.upload_folder)
        self.upload_button.pack(side=tk.LEFT, padx=(5,20), pady=10)

        self.back_button = tk.Button(self.button_frame, text="<  Back", command=self.go_back, state=tk.DISABLED)
        self.back_button.pack(side=tk.LEFT, padx=2, pady=10)

        self.next_button = tk.Button(self.button_frame, text="Next  >", command=self.go_next, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=2, pady=10)


        # # Create the upload buttons
        # self.upload_button_a = tk.Button(self.button_frame, text="Upload Images", command=self.upload_images)
        # self.upload_button_a.pack(side=tk.LEFT, padx=5, pady=10)

        # self.upload_button_b = tk.Button(self.button_frame, text="Upload Image B", command=self.upload_image_b)
        # self.upload_button_b.pack(side=tk.LEFT, padx=5, pady=10)

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
        self.change_var = tk.BooleanVar()
        self.changebox = tk.Checkbutton(self.button_frame, text="Change?", variable=self.change_var, command=self.changebox_changed)
        self.changebox.pack(side=tk.LEFT, padx=(35, 0))




        # # Create buttons to hide and unhide rectangles
        # self.hide_button = tk.Button(root, text="Hide Rectangles", command=self.hide_rects)
        # self.hide_button.pack()

        # self.unhide_button = tk.Button(root, text="Unhide Rectangles", command=self.unhide_rects)
        # self.unhide_button.pack()


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
        self.root.bind("<Control-+>", self.zoom_in)
        self.root.bind("<Control-_>", self.zoom_out)        
        self.root.bind("<Control-)>", self.zoom_reset)                

        self.root.bind("<Control-h>", self.hide_rects)                
        # self.root.bind("<Control-u>", self.unhide_rects)                        

        self.root.bind("<Control-m>", self.go_next)                
        self.root.bind("<Control-n>", self.go_back)   

        self.canvas.bind("<Button-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)    
        self.canvas.bind("<Shift-Button-1>", self.delete_box)


        # Initialize drawing state
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.rectangles = []

        self.clear_boxes()



    def upload_images(self):
        print(self.label.cget("text"))
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

        self.new_img = 1

        self.clear_boxes()  # Clear existing rectangles


        # upload rectangles ---------------------------------------------
        # jsons = 'boxes.json' #  [img for img in files if img.endswith('.json')]
        self.boxes_path = os.path.join(self.label.cget("text"), 'boxes.json')
        if not os.path.exists(self.boxes_path):  # create a dummy json
            with open(self.boxes_path, 'w') as f:
                # boxes = {}
                json.dump({}, f)

        with open(self.boxes_path, 'r') as f:
            boxes = json.load(f)
            for box in boxes.values():
                rect = [box["xl"], box["yl"], box["xr"], box["yr"]]
                self.rectangles.append(rect)
                # self.canvas.create_rectangle(*self.scale_coordinates(rect), outline='yellow')      
            


        # upload change ---------------------------------------------
        self.change_path = os.path.join(self.label.cget("text"), 'change.json')
        if not os.path.exists(self.change_path):  # create a dummy json
            with open(self.change_path, 'w') as f:
                json.dump({},f)

        with open(self.change_path, 'r') as f:
            change = json.load(f)
            # breakpoint()
            self.change_var = change.get('change', False)
            if self.change_var or len(self.rectangles)>0:
                self.changebox.select()
            else:
                self.changebox.deselect()



        self.update_image()



    def toggle_image(self, event=None):
        # if self.image_a and self.image_b:
            # self.current_image = self.image_b if self.current_image == self.image_a else self.image_a
        # print(self.current_path, self.image_path_1, self.image_path_2, self.new_img)
        if self.current_path == self.image_path_1:
            self.current_image = self.image_b
            self.current_path = self.image_path_2
            self.label_2.config(fg="blue")
            self.label_1.config(fg="black")
            # print('X')
        else:
            self.current_image = self.image_a
            self.current_path = self.image_path_1
            self.label_1.config(fg="blue")
            self.label_2.config(fg="black")
            # print('Y')
        self.update_image()

        if not self.hidden_state:
            self.clear_canvas()


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

            # print(self.zoom_factor, self.pan_offset_x, self.pan_offset_y)


            self.delete_box_size = self.checkbox_var.get()
            if self.delete_box_size:
                self.rectangles = [rect for rect in self.rectangles if (rect[2] - rect[0]) * (rect[3] - rect[1]) >= self.box_size]
            

            for rect in self.rectangles:
                # do the filtering 
                # print(rect)
                try:
                    if not rect: continue
                    rect_area  = (rect[2] - rect[0]) * (rect[3] - rect[1])
                    if rect_area >= self.box_size:
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
            # Hide all rectangles
            for rect in self.rectangles:
                try:
                     self.canvas.itemconfig(rect, state=tk.HIDDEN)
                except:
                    pass
            self.clear_canvas()
            self.hidden_state = False

        else:
            for rect in self.rectangles:
                try:
                    self.canvas.itemconfig(rect, state=tk.NORMAL)            
                except:
                    pass
            self.update_image()
            self.hidden_state = True


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
        try:
            # if self.boxes_path:
            boxes = {f"box_{i}": {"xl": int(coords[0]), "yl": int(coords[1]), "xr": int(coords[2]), "yr": int(coords[3])} for i, coords in enumerate(self.rectangles) if coords}
            with open(self.boxes_path , 'w') as f:
                json.dump(boxes, f)

            # messagebox.showinfo("Information", "Boxes were saved in boxes.json")                
            
        except Exception as e:
            messagebox.showinfo('Error', e)                


    def upload_boxes(self):
        with open(self.boxes_path, 'r') as f:
            boxes = json.load(f)
        self.clear_boxes()  # Clear existing rectangles
        for box in boxes.values():
            rect = [box["xl"], box["yl"], box["xr"], box["yr"]]
            self.rectangles.append(rect)
            # self.canvas.create_rectangle(*self.scale_coordinates(rect), outline='yellow')

        self.update_image()                


    def upload_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir()]

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
        if self.subfolders:
            current_folder = self.subfolders[self.current_index]
            self.label.config(text=current_folder)
        else:
            self.label.config(text="No subfolders found")


    def update_buttons(self):
        self.back_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)




    def update_global_number(self, event):
        try:
            # Get the content from the Entry widget and convert it to a float
            self.box_size = float(self.entry.get())
            # messagebox.showinfo("Global Number Updated", f"Global number updated to: {global_number}")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")

        self.update_image()
        self.canvas.focus_set()


    def checkbox_changed(self):
        self.update_image()
        self.canvas.focus_set()


    def changebox_changed(self):
        # if self.boxes_path:
        # with open(self.change_path , 'r') as f:
        #     data = json.load(f)

        # add the change info 
        # breakpoint()
        # print(self.change_var)
        self.change_var = not self.change_var
        data = {}
        data['change'] = self.change_var #.get()
        with open(self.change_path , 'w') as f:
            json.dump(data, f)





    def show_shortcuts(self):
        shortcuts = [
            'Shift + H : Hide and unhide boxes',
            'Shift + M : Move to the next image',
            'Shift + N: Move to the previous image',
            'Shift + + : Zoom in',
            'Shift + - : Zoom out',
            'Shift + ) : Reset zoom',
            'Space bar : Toggle between two images',
            'Middle mouse button and drag : Drag the image around',
            'Right click : Remove the box at the mouse location',
            'Left click and drag : Draw a box'
        ]
        
        shortcuts_message = "\n".join(shortcuts)
        messagebox.showinfo("Shortcuts", shortcuts_message)



if __name__ == "__main__":
    root = tk.Tk()


    app = ImageToggleApp(root) #, img1, img2)
    root.mainloop()
