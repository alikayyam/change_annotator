import tkinter as tk
from tkinter import messagebox

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Checkbox Example")

        # Create a frame to hold widgets
        self.frame = tk.Frame(root)
        self.frame.pack(padx=10, pady=10)

        # Create a label
        self.label = tk.Label(self.frame, text="Check the box if you agree:")
        self.label.pack(side=tk.LEFT, padx=(0, 5))

        # Create a BooleanVar to store the state of the checkbox
        self.checkbox_var = tk.BooleanVar()

        # Create a checkbox and bind it to the BooleanVar
        self.checkbox = tk.Checkbutton(self.frame, text="I Agree", variable=self.checkbox_var, command=self.checkbox_changed)
        self.checkbox.pack(side=tk.LEFT, padx=(5, 0))

        # Create a submit button
        self.submit_button = tk.Button(root, text="Submit", command=self.on_submit)
        self.submit_button.pack(pady=10)

    def checkbox_changed(self):
        if self.checkbox_var.get():
            print("Checkbox is checked")
        else:
            print("Checkbox is unchecked")

    def on_submit(self):
        if self.checkbox_var.get():
            messagebox.showinfo("Submit", "Thank you for agreeing!")
        else:
            messagebox.showwarning("Submit", "Please agree to continue.")

# Create the main window
root = tk.Tk()
app = App(root)

# Run the Tkinter event loop
root.mainloop()
