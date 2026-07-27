# Insect Image Dataset Loader

This project provides a PyTorch `Dataset` implementation for loading and processing insect images with bounding box annotations. 

## Overview

The `InsectDataset` class (in `test.py`) allows you to load insect images and their corresponding annotations for machine learning tasks. It handles:
- Reading bounding box annotations from CSV files.
- Automatically loading the corresponding JPEG images.
- Cropping the images to their specified bounding boxes.
- Encoding class labels using `sklearn.preprocessing.LabelEncoder`.
- Applying PyTorch transformations (e.g., converting to tensors and normalizing).

## Project Structure

- `test.py`: Contains the main `InsectDataset` class which inherits from `torch.utils.data.Dataset`. It also includes a sample test script that demonstrates how to instantiate the dataset.
- `ShadowGraph/`: Directory containing project data (e.g., MorphoCluster image samples and annotation results).

## Requirements

The project requires the following Python packages:
- `torch`
- `torchvision`
- `pandas`
- `Pillow` (PIL)
- `scikit-learn`

## Usage

You can use the dataset by providing the paths to your annotation CSVs and your image folder.

```python
from test import InsectDataset
from torchvision import transforms

# Define paths to your data
dataset_path = 'path/to/annotations' # Folder containing .csv files with bbox_x, bbox_y, bbox_w, bbox_h, and class_name
image_path = 'path/to/images'        # Folder containing corresponding .jpg images

# Create the dataset
dataset = InsectDataset(dataset_path=dataset_path, image_path=image_path)

# Access an item
img, label, path = dataset[0]

print(f"Image shape: {img.shape}, Label: {label}")
```

Each CSV file in the `dataset_path` should correspond to an image and contain columns such as `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, and `class_name`.
