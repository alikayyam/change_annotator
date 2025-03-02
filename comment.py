import tkinter as tk
from tkinter import Toplevel, Text, Button

class CommentApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Main Window")
        
        self.open_button = Button(master, text="Open Comment Window", command=self.open_comment_window)
        self.open_button.pack(pady=20)
        
    def open_comment_window(self):
        # Create a new window
        self.comment_window = Toplevel(self.master)
        self.comment_window.title("Comment Window")
        
        # Add a text widget for the user to type the comment
        self.comment_text = Text(self.comment_window, wrap='word', width=50, height=10)
        self.comment_text.pack(padx=10, pady=10)
        
        # Add a submit button to close the comment window and print the comment
        self.submit_button = Button(self.comment_window, text="Submit", command=self.submit_comment)
        self.submit_button.pack(pady=10)
        
    def submit_comment(self):
        # Get the comment from the text widget
        comment = self.comment_text.get("1.0", tk.END).strip()
        
        # Print the comment (you can modify this part to handle the comment as needed)
        print("Comment submitted:", comment)
        
        # Close the comment window
        self.comment_window.destroy()

# Create the main window
root = tk.Tk()
app = CommentApp(root)
root.mainloop()
