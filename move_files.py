import os
import shutil

# Define the path to the root folder
root_folder = r"C:\Users\ali.borji\Eyenuk Dropbox\SharedInternal\Clinical\Dennis\ChangeLabel_20240514"

# Get the list of all items in the root folder
for item in os.listdir(root_folder):
    item_path = os.path.join(root_folder, item)
    
    # Check if the item is a folder (subfolder)
    if os.path.isdir(item_path):
        subfolder = item
        
        # Paths to the existing boxes.json and change.json in the subfolder
        boxes_json_path = os.path.join(item_path, "boxes.json")
        old_boxes_json_path = os.path.join(item_path, "old_boxes.json")
        change_json_path = os.path.join(item_path, "change.json")
        
        
        # If change.json exists, remove it
        if os.path.exists(change_json_path):
            os.remove(change_json_path)
        
        # Path to the corresponding flat JSON file in the root folder
        flat_json_file = f"{subfolder}boxes.json"
        flat_json_path = os.path.join(root_folder, flat_json_file)
        
        # If the flat JSON file exists, move it to the subfolder and rename it to boxes.json
        if os.path.exists(flat_json_path):

            # If boxes.json exists, rename it to old_boxes.json
            if os.path.exists(boxes_json_path):
                os.rename(boxes_json_path, old_boxes_json_path)

            # write the new one
            shutil.move(flat_json_path, boxes_json_path)

print("Operation completed successfully.")
