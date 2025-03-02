import tkinter as tk
from tkinter import filedialog
import os

class FolderNavigator:
    def __init__(self, root):
        self.root = root
        self.root.title("Folder Navigator")

        self.current_index = 0
        self.subfolders = []

        # Create and pack the label to show the current folder
        self.label = tk.Label(root, text="No folder selected", wraplength=400)
        self.label.pack(pady=10)

        # Create and pack the buttons
        self.upload_button = tk.Button(root, text="Upload Root Folder", command=self.upload_folder)
        self.upload_button.pack(pady=10)

        self.back_button = tk.Button(root, text="Back", command=self.go_back, state=tk.DISABLED)
        self.back_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.next_button = tk.Button(root, text="Next", command=self.go_next, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT, padx=20, pady=10)

    def upload_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
            self.current_index = 0
            self.update_label()
            self.update_buttons()

    def go_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_label()
            self.update_buttons()

    def go_next(self):
        if self.current_index < len(self.subfolders) - 1:
            self.current_index += 1
            self.update_label()
            self.update_buttons()

    def update_label(self):
        if self.subfolders:
            current_folder = self.subfolders[self.current_index]
            self.label.config(text=current_folder)
        else:
            self.label.config(text="No subfolders found")

    def update_buttons(self):
        self.back_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.subfolders) - 1 else tk.DISABLED)

# Create the main window
root = tk.Tk()
app = FolderNavigator(root)

# Run the application
root.mainloop()
