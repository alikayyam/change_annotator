# import tkinter as tk
# from tkinter import messagebox

# def get_number():
#     try:
#         # Get the content from the Entry widget and convert it to a float
#         number = float(entry.get())
#         messagebox.showinfo("Input Number", f"You entered: {number}")
#     except ValueError:
#         messagebox.showerror("Invalid Input", "Please enter a valid number")

# # Create the main window
# root = tk.Tk()
# root.title("Number Input")

# # Create a Label to instruct the user
# label = tk.Label(root, text="Enter a number:")
# label.pack(pady=10)

# # Create an Entry widget for number input
# entry = tk.Entry(root)
# entry.pack(pady=5)

# # Create a Button to get the number
# button = tk.Button(root, text="Submit", command=get_number)
# button.pack(pady=10)

# # Run the Tkinter event loop
# root.mainloop()

import tkinter as tk
from tkinter import messagebox

def get_number():
    try:
        # Get the content from the Entry widget and convert it to a float
        number = float(entry.get())
        messagebox.showinfo("Input Number", f"You entered: {number}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number")

# Create the main window
root = tk.Tk()
root.title("Number Input")

# Create a Label to instruct the user and place it in the first row, first column
label = tk.Label(root, text="Enter a number:")
label.grid(row=0, column=0, padx=10, pady=10)

# Create an Entry widget for number input and place it in the first row, second column
entry = tk.Entry(root)
entry.grid(row=0, column=1, padx=10, pady=10)

# Create a Button to get the number and place it in the second row, spanning both columns
button = tk.Button(root, text="Submit", command=get_number)
button.grid(row=1, column=0, columnspan=2, pady=10)

# Run the Tkinter event loop
root.mainloop()
