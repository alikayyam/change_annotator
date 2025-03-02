import tkinter as tk

class CanvasApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack()

        # Create some rectangles
        self.rects = []
        self.rects.append(self.canvas.create_rectangle(50, 50, 150, 150, fill='blue', outline='black'))
        self.rects.append(self.canvas.create_rectangle(200, 50, 300, 150, fill='green', outline='black'))

        # Create buttons to hide and unhide rectangles
        self.hide_button = tk.Button(root, text="Hide Rectangles", command=self.hide_rects)
        self.hide_button.pack()

        self.unhide_button = tk.Button(root, text="Unhide Rectangles", command=self.unhide_rects)
        self.unhide_button.pack()

        # Bind Shift+H to hide rectangles
        self.root.bind('<Shift-H>', self.hide_rects)
        self.root.bind('<Shift-U>', self.unhide_rects)

    def hide_rects(self, event=None):
        # Hide all rectangles
        for rect in self.rects:
            self.canvas.itemconfig(rect, state='hidden')

    def unhide_rects(self):
        # Unhide all rectangles
        for rect in self.rects:
            self.canvas.itemconfig(rect, state='normal')

# Create the main window
root = tk.Tk()
app = CanvasApp(root)

# Run the application
root.mainloop()
