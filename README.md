
# Image Pair Viewer and Annotator

This tool allows you to upload and view a pair of images, toggle between them, interact with the images by panning and zooming, and annotate the images by drawing bounding boxes. Once you've drawn the bounding boxes, the results can be saved as a JSON file, which includes the details of the annotations made on the images.

## Features

- **Upload Image Pair**: Users can upload two images that they wish to compare side-by-side.
- **Toggle Between Images**: Easily switch between the two images in the pair to view them separately.
- **Pan and Zoom**: Navigate through the images with smooth panning and zooming capabilities. You can zoom in for a closer look or zoom out to get an overview of the image.
- **Draw Bounding Boxes**: Annotate the images by drawing bounding boxes. You can create multiple boxes, adjust their positions, and resize them as needed.
- **Save Annotations as JSON**: After drawing bounding boxes, save your annotations in a JSON file. This file will store the coordinates of the boxes, along with any other relevant metadata you provide.

## Installation

To use this tool, simply clone this repository and open it in your browser. Follow the steps below:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-repository/image-pair-viewer.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd image-pair-viewer
   ```

3. Open the `index.html` file in your browser to start using the application.

## Usage

### Upload Image Pair

Once the application is open, you will see an interface that allows you to upload two images. Use the "Upload Images" button to choose your image files. Make sure you select two images for comparison.

### Toggle Between Images

After uploading your image pair, you can easily toggle between the two images. This feature is helpful when comparing side-by-side images, such as before-and-after images, or images with different filters applied.

### Pan and Zoom

You can interact with the images by panning and zooming. Here’s how:

- **Zoom**: Use your mouse scroll wheel or pinch to zoom in and out of the image.
- **Pan**: Click and drag to move around the image and explore different areas.

This allows for a detailed view and enables users to examine specific parts of the image with precision.

### Draw Bounding Boxes

To annotate the images, you can draw bounding boxes. Here’s how:

- **Drawing a Box**: Click and drag on the image to create a bounding box.
- **Resize or Move a Box**: Click and drag the edges of the box to resize it, or click and drag the entire box to move it to a new position.
- **Multiple Boxes**: You can draw as many bounding boxes as you need on each image.
  
Bounding boxes are useful for marking regions of interest, such as objects or areas that you want to label.

### Save Annotations as JSON

Once you have drawn your bounding boxes and made your annotations, you can save the results as a JSON file. This file will contain the coordinates of each bounding box along with any additional metadata (such as labels or categories) that you specify.

To save the annotations:

1. After drawing your boxes, click the "Save Annotations" button.
2. A JSON file will be generated and automatically downloaded. This file contains the details of all the boxes you’ve drawn, including:
   - The image filenames
   - The coordinates of each bounding box (x, y, width, height)
   - Optional labels or other metadata you provide

This file can then be used for further processing, such as training machine learning models or keeping track of your annotations for documentation purposes.

## Example JSON Structure

Here’s an example of what the JSON output might look like after saving your annotations:

```json
{
  "image_1": {
    "file_name": "image_1.jpg",
    "boxes": [
      {
        "label": "Object A",
        "coordinates": {
          "x": 50,
          "y": 100,
          "width": 150,
          "height": 200
        }
      },
      {
        "label": "Object B",
        "coordinates": {
          "x": 200,
          "y": 250,
          "width": 100,
          "height": 150
        }
      }
    ]
  },
  "image_2": {
    "file_name": "image_2.jpg",
    "boxes": [
      {
        "label": "Object A",
        "coordinates": {
          "x": 60,
          "y": 120,
          "width": 160,
          "height": 210
        }
      }
    ]
  }
}
```

## Requirements

This tool works directly in the browser and does not require any server-side configuration or installation. It relies on the following technologies:

- HTML5
- JavaScript
- CSS3

## Contributing

Feel free to open issues or pull requests if you'd like to contribute improvements to the project.

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/image-pair-viewer.git
   ```
3. Create a new branch:
   ```bash
   git checkout -b my-new-feature
   ```
4. Make your changes
5. Commit your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
6. Push to your fork:
   ```bash
   git push origin my-new-feature
   ```
7. Open a pull request!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
