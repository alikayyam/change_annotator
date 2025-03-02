import tkinter as tk

def on_variable_change(*args):
    # Callback function to handle variable changes
    if some_variable.get() == 1:
        checkbox.select()  # Check the checkbox
    else:
        checkbox.deselect()  # Uncheck the checkbox

def toggle_checkbox():
    # Function to toggle checkbox state
    if checkbox_var.get() == 1:
        checkbox.deselect()  # Uncheck the checkbox
    else:
        checkbox.select()  # Check the checkbox

# Create a Tkinter window
root = tk.Tk()
root.title("Checkbox Example")

# Simulating a variable that changes
some_variable = tk.IntVar()
some_variable.set(0)  # Initial value



# Create a checkbox widget
checkbox_var = tk.IntVar()
checkbox = tk.Checkbutton(root, text="Check me", variable=checkbox_var)
checkbox.pack(pady=10)

# Button to toggle checkbox state
toggle_button = tk.Button(root, text="Toggle Checkbox", command=toggle_checkbox)
toggle_button.pack(pady=5)

# Function to update checkbox state when some_variable changes
some_variable.trace_add("write", on_variable_change)

# Run the Tkinter event loop
root.mainloop()

# Example of changing the variable elsewhere in your code
# Simulating a change after 3 seconds
import time
time.sleep(3)
some_variable.set(1)
