import tkinter as tk
from tkinter import ttk

def on_combobox_select(event):
    # Change focus to the canvas when an item is selected in the combobox
    canvas.focus_set()

def draw_on_canvas(event):
    # Draw a circle on the canvas at the mouse position
    x, y = event.x, event.y
    canvas.create_oval(x-10, y-10, x+10, y+10, fill='blue')

# Create the main window
root = tk.Tk()
root.title("Focus Example")

# Create a list of options for the combobox
options = ["Option 1", "Option 2", "Option 3"]

# Create the Combobox widget
combo = ttk.Combobox(root, values=options)
combo.pack(padx=10, pady=10)

# Bind the selection event to change focus
combo.bind("<<ComboboxSelected>>", on_combobox_select)

# Create a Canvas widget
canvas = tk.Canvas(root, width=400, height=300, bg='lightgrey')
canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Bind mouse click event to draw on the canvas
canvas.bind("<Button-1>", draw_on_canvas)

# Run the application
root.mainloop()
