import tkinter as tk

class CanvasApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=300, bg='white')
        self.canvas.pack()

        # Create some canvas items
        self.line = self.canvas.create_line(10, 10, 200, 200, fill='blue')
        self.rect = self.canvas.create_rectangle(50, 50, 150, 100, fill='green')
        self.oval = self.canvas.create_oval(200, 50, 300, 150, fill='red')

        # Get and print coordinates of the items
        self.print_coords()

        # Modify the coordinates of the items
        self.move_items()

        # Print the new coordinates
        self.print_coords()

    def print_coords(self):
        line_coords = self.canvas.coords(self.line)
        rect_coords = self.canvas.coords(self.rect)
        oval_coords = self.canvas.coords(self.oval)
        print(f"Line coords: {line_coords}")
        print(f"Rectangle coords: {rect_coords}")
        print(f"Oval coords: {oval_coords}")

    def move_items(self):
        # Move the line
        self.canvas.coords(self.line, 20, 20, 250, 250)
        
        # Move the rectangle
        self.canvas.coords(self.rect, 60, 60, 170, 120)
        
        # Move the oval
        self.canvas.coords(self.oval, 220, 70, 320, 170)

# Create the main window
root = tk.Tk()
app = CanvasApp(root)
root.mainloop()
