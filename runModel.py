import os
import cv2
import xml.etree.ElementTree as ET
from genSensorModel import GenSensorModel

# Function to parse VOC annotations
def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels

# Path to directory containing annotations and JPEG images
data_dir = "../NTNLP/NTLNP/voc_day"
annotations_dir = os.path.join(data_dir, "Annotations")
images_dir = os.path.join(data_dir, "JPEGImages")

# Instantiate the model
model = GenSensorModel()

# Iterate over each image and annotation
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(images_dir, filename)
        annotation_path = os.path.join(annotations_dir, filename[:-4] + ".xml")

        # Read the image
        image = cv2.imread(image_path)

        # Parse the VOC annotation
        boxes, labels = parse_voc_annotation(annotation_path)

        # Run the model on the image
        predictions = model.run_model(image)

        # Optionally, visualize the results by drawing the predicted bounding boxes on the image
        if predictions is not None:
            for bbox in predictions:
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # Display or save the result image as needed
        cv2.imshow("Result", image)
        cv2.waitKey(0)  # Press any key to move to the next image

cv2.destroyAllWindows()  # Close all windows when done
