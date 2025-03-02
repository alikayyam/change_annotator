from PIL import Image, ImageOps, ImageEnhance

def overlay_mask(image_path, mask_path, output_path, mask_color=(255, 0, 0, 128)):
    # Load the image and the mask
    image = Image.open(image_path).convert("RGBA")  # Convert image to RGBA to support transparency
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale ('L' mode)

    # Ensure mask is binary (0 or 255)
    mask = mask.point(lambda p: p > 128 and 255) 

    # Create an RGBA image for the mask with the specified color and apply the mask
    mask_colored = Image.new("RGBA", image.size, mask_color)
    mask_colored.putalpha(mask)  # Apply the mask to the colored image

    # Overlay the mask on the image
    result = Image.alpha_composite(image, mask_colored)

    # Save the result
    result.save(output_path, "PNG")

# Example usage
image_path = r"C:\Users\ali.borji\Documents\gui\data3\0000109_t110_i007\0000109_t110_i007.jpg"
mask_path = r"C:\Users\ali.borji\Documents\gui\data3\0000109_t110_i007\hemorrhages_0000109_t110_i007.png"
output_path = r"C:\Users\ali.borji\Documents\gui\data3\0000109_t110_i007\xx.png"
overlay_mask(image_path, mask_path, output_path)
