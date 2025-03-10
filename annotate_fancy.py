import tkinter as tk             
from PIL import Image, ImageTk  
import numpy as np                                

class PanZoomCanvas(tk.Frame):
    def __init__(self, master,canvas_w,canvas_h):
        super().__init__(master)
        self.pil_image = None   # Image data to be displayed

        self.zoom_cycle = 0

        self.create_widget(canvas_w,canvas_h) # Create canvas
        
        # Initial affine transformation matrix
        self.reset_transform()
 
    # Define the create_widget method.
    def create_widget(self,width,height):
        # Canvas
        self.canvas = tk.Canvas(self.master, background="black", width = width,height = height)
        self.canvas.pack() 

        # Controls
        self.master.bind("<Button-1>", self.mouse_down_left)                   # MouseDown
        self.master.bind("<B1-Motion>", self.mouse_move_left)                  # MouseDrag
        self.master.bind("<Double-Button-1>", self.mouse_double_click_left)    # MouseDoubleClick
        self.master.bind("<MouseWheel>", self.mouse_wheel)                     # MouseWheel


    def set_image(self, filename):
        '''To open an image file'''
        if not filename:
            return
        # PIL.Image
        self.pil_image = Image.open(filename)
        # Set the affine transformation matrix to display the entire image.
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        # To display the image
        self.draw_image(self.pil_image)

    # -------------------------------------------------------------------------------
    # Mouse events
    # -------------------------------------------------------------------------------
    def mouse_down_left(self, event):
        self.__old_event = event

    def mouse_move_left(self, event):
        if (self.pil_image == None):
            return
        
        self.translate(event.x - self.__old_event.x, event.y - self.__old_event.y)
        self.redraw_image()
        self.__old_event = event



    def mouse_double_click_left(self, event):
        if self.pil_image == None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw_image() 

    def mouse_wheel(self, event):
        if self.pil_image == None:
            return

        if (event.delta < 0):
            if self.zoom_cycle <= 0:
                return
            # Rotate upwards and shrink
            self.scale_at(0.8, event.x, event.y)
            self.zoom_cycle -= 1
        else:
            if self.zoom_cycle >= 9:
                return
            #  Rotate downwards and enlarge
            self.scale_at(1.25, event.x, event.y)
            self.zoom_cycle += 1
    
        self.redraw_image() # Refresh
        
    # -------------------------------------------------------------------------------
    # Affine Transformation for Image Display
    # -------------------------------------------------------------------------------

    def reset_transform(self):
        self.mat_affine = np.eye(3) # 3x3の単位行列

    def translate(self, offset_x, offset_y,zoom = False):
        mat = np.eye(3) # 3x3 identity matrix
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)
        # Get the current canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get the current scale
        scale = self.mat_affine[0, 0]
        max_y = scale * 3072
        max_x = scale * 4096
        self.mat_affine = np.dot(mat, self.mat_affine)

        if not zoom:
            if abs(self.mat_affine[0,2]) > abs(max_x-canvas_width):
                self.mat_affine[0,2] = -(max_x-canvas_width)
            if abs(self.mat_affine[1,2]) > abs(max_y-canvas_height):
                self.mat_affine[1,2] = -(max_y-canvas_height)

        if self.mat_affine[0, 2] > 0.0:
            self.mat_affine[0, 2] = 0.0
        if self.mat_affine[1,2] > 0.0:
            self.mat_affine[1,2]  = 0.0

    def scale(self, scale:float):
        mat = np.eye(3) # 3x3 identity matrix

        mat[0, 0] = scale
        mat[1, 1] = scale
        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale_at(self, scale:float, cx:float, cy:float):

        # Translate to the origin
        self.translate(-cx, -cy, True)
        # Scale
        self.scale(scale)
        # Restore
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):

        # Update canvas object and get size
        self.master.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if (image_width * image_height <= 0) or (canvas_width * canvas_height <= 0):
            return

        # Initialization of affine transformation
        self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0
        if (canvas_width * image_height) > (image_width * canvas_height):
            # The widget is horizontally elongated (resizing the image vertically)
            scale = canvas_height / image_height
            # Align the remaining space to the center by offsetting horizontally
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            # The widget is vertically elongated (resizing the image horizontally)
            scale = canvas_width / image_width
            # Align the remaining space to the center by offsetting vertically
            offsety = (canvas_height - image_height * scale) / 2

        # Scale
        self.scale(scale)
        # Align the remaining space to the center
        self.translate(offsetx, offsety)
        self.zoom_cycle = 0

    def to_image_point(self, x, y):
        '''Convert coordinates from the canvas to the image'''
        if self.pil_image == None:
            return []
        # Convert coordinates from the image to the canvas by taking the inverse of the transformation matrix.
        mat_inv = np.linalg.inv(self.mat_affine)
        image_point = np.dot(mat_inv, (x, y, 1.))
        if  image_point[0] < 0 or image_point[1] < 0 or image_point[0] > self.pil_image.width or image_point[1] > self.pil_image.height:
            return []

        return image_point

    # -------------------------------------------------------------------------------
    # Drawing 
    # -------------------------------------------------------------------------------

    def draw_image(self, pil_image):
        
        if pil_image == None:
            return

        self.pil_image = pil_image

        # Canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate the affine transformation matrix from canvas to image data
        # (Calculate the inverse of the display affine transformation matrix)
        mat_inv = np.linalg.inv(self.mat_affine)

        # Convert the numpy array to a tuple for affine transformation
        affine_inv = (
            mat_inv[0, 0], mat_inv[0, 1], mat_inv[0, 2],
            mat_inv[1, 0], mat_inv[1, 1], mat_inv[1, 2]
        )

        # Apply affine transformation to the PIL image data
        dst = self.pil_image.transform(
            (canvas_width, canvas_height),  # Output size
            Image.AFFINE,   # Affine transformation
            affine_inv,     # Affine transformation matrix (conversion matrix from output to input)
            Image.NEAREST   # Interpolation method, nearest neighbor
        )

        im = ImageTk.PhotoImage(image=dst)

        # Draw the image
        item = self.canvas.create_image(
            0, 0,           # Image display position (top-left coordinate)
            anchor='nw',    # Anchor, top-left is the origin
            image=im        # Display image data
        )
        self.image = im

    def redraw_image(self):
        '''Redraw the image'''
        if self.pil_image == None:
            return
        self.draw_image(self.pil_image)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('1200x900')
    app = PanZoomCanvas(master=root,canvas_w = 1024,canvas_h = 768)
    app.canvas.config(bg = 'grey')
    app.set_image('0000109_t110_i006.jpg')
    app.mainloop()
