import tkinter as tk

def on_ctrl_key(event):
    print("Ctrl + Key pressed:", event.keysym)

root = tk.Tk()
root.geometry("400x300")

frame = tk.Frame(root, width=400, height=300, bg="lightblue")
frame.pack()

# Bind Ctrl key combinations to the frame
root.bind("<Control-KeyPress-a>", on_ctrl_key)  # Ctrl + A
root.bind("<Control-KeyPress-s>", on_ctrl_key)  # Ctrl + S
root.bind("<Control-KeyPress-c>", on_ctrl_key)  # Ctrl + C

root.mainloop()



# DRS_merged,,/mnt/TechTeamDrive/RetinalDatasets/STANDARDIZED/drsplus_merged/images/,Research/Python/datasets/DRS_merged/,classification,"dr, csme",20495,19882,,"iCare DRSplus, eidon",DRSplus train set,