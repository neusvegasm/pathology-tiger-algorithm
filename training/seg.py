import json
from PIL import Image, ImageDraw

# Load the COCO annotations
with open('tiger-coco.json') as f:
    coco_data = json.load(f)


def draw_lymphocyte(draw, center, rB, rM):
    # Draw membrane (larger circle)
    # I need the membrane to be labeled as 1 and the cell body as 2

    draw.ellipse([(center[0] - rM, center[1] - rM), (center[0] + rM, center[1] + rM)], fill=2, outline=2)  # gray for membrane

    # Draw cell body (smaller circle)
    draw.ellipse([(center[0] - rB, center[1] - rB), (center[0] + rB, center[1] + rB)], fill=1)  # white for cell body

# Convert micrometers to pixels
scale_factor = 0.5  # 0.5 um/px
rB_pixels = int(2.4 / scale_factor)  # Cell body radius in pixels
rM_pixels = int(2.88 / scale_factor)  # Membrane radius in pixels

# Process each image
for image_info in coco_data['images']:
    image_path = image_info['file_name']
    image = Image.open(image_path)
    mask = Image.new('L', image.size, 0)  # Create a new black mask
    draw = ImageDraw.Draw(mask)

    # Get annotations for this image
    image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
    
    for annotation in image_annotations:
        # Assume annotation['bbox'] contains the center coordinates
        center_x = annotation['bbox'][0] + annotation['bbox'][2] / 2
        center_y = annotation['bbox'][1] + annotation['bbox'][3] / 2
        draw_lymphocyte(draw, (center_x, center_y), rB_pixels, rM_pixels)

    # Save the modified image as grayscale

    # mask.save('labels/' + image_info['file_name'][9:])

print("Segmentation masks for lymphocytes have been generated and saved.")