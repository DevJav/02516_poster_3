# %%
from sklearn.metrics import average_precision_score
from torchvision.ops import nms
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os
import xml.etree.ElementTree as ET
import cv2

# %%
class PotholeDatasetFromProposals(Dataset):
    def __init__(self, file_list, img_dir, proposals, transform=None, target_ratio=7.0, apply_ratio=False, test=False):
        """
        Initialize the Pothole dataset using only the proposals JSON.
        
        Args:
            file_list (list): List of image filenames.
            img_dir (str): Directory where images are stored.
            proposals (dict): Proposed bounding boxes and labels for each image.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.transform = transform

        if apply_ratio:
            # Apply ratio to achieve some class balance
            self.samples = []
            for img_name in file_list:
                background_proposals = [proposal for proposal in proposals.get(img_name, []) if proposal['label'] == 'background']
                pothole_proposals = [proposal for proposal in proposals.get(img_name, []) if proposal['label'] == 'pothole']

                # Sample pothole proposals to match the background proposals
                num_background = len(background_proposals)
                num_potholes = len(pothole_proposals)

                if num_potholes == 0:
                    if test:
                        continue
                    boxes = self._add_from_xml(img_name)
                    pothole_proposals = boxes
                    num_potholes = len(boxes)
                
                #required background count should be the compliment of the ratio, so if there are 3 potholes, and the ratio is 0.3 then the required background count should be 7
                required_background_count = int(num_potholes * (( 1 / target_ratio ) - 1 ))
                required_background_count = min(num_background, required_background_count)

                for proposal in pothole_proposals:
                    self.samples.append({
                        "img_name": img_name,
                        "bbox": proposal,
                    })
                
                #get random background proposals to match the required count                
                background_boxes = random.sample(background_proposals, required_background_count)
                for proposal in background_boxes:
                    self.samples.append({
                        "img_name": img_name,
                        "bbox": proposal,
                    })

        else:
            # Flatten the dataset to treat each bounding box as an independent sample
            self.samples = []
            for img_name in file_list:
                for proposal in proposals.get(img_name, []):
                    self.samples.append({
                        "img_name": img_name,
                        "bbox": proposal,
                    })

    def __len__(self):
        return len(self.samples)
    
    def _add_from_xml(self, img_name):
        # Same as before, no modification needed
        boxes = []
        xml_file = img_name.replace('.jpg', '.xml')
        tree = ET.parse(os.path.join(self.img_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall("object"):
            if obj.find("name").text == "pothole":
                bbox = obj.find("bndbox")
                xmin, ymin, xmax, ymax = map(int, (bbox.find("xmin").text, bbox.find("ymin").text, bbox.find("xmax").text, bbox.find("ymax").text))
                box = {
                    "x": xmin,
                    "y": ymin,
                    "width": xmax - xmin,
                    "height": ymax - ymin,
                    "label": "pothole",
                }
                boxes.append(box)
        return boxes
        

    def __getitem__(self, idx, get_bbox=True):
        sample = self.samples[idx]
        img_name = sample["img_name"]
        bbox = sample["bbox"]

        if not self.samples[idx].get("crop"):      
            # Load and crop the image to the bounding box region
            image_path = os.path.join(self.img_dir, img_name)
            image = Image.open(image_path).convert("RGB")

            # Original bounding box coordinates
            x, y, width, height = (
                bbox["x"],
                bbox["y"],
                bbox["width"],
                bbox["height"]
            )

            # Crop the image to the bounding box
            cropped_image = image.crop((x, y, x + width, y + height))
            self.samples[idx]["crop"] = cropped_image
        
        cropped_image = self.samples[idx]["crop"]
        # Apply transformations to the cropped region
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        # Determine label
        label = 0 if bbox['label'] == "background" else 1  # 0 for "background", 1 for "pothole"
        
        if get_bbox:
            return cropped_image, label, torch.tensor([bbox["x"], bbox["y"], bbox["width"], bbox["height"]]), img_name
        else:
            return cropped_image, label

# %%
import json
import datetime

notebook_dir = os.getcwd()
print("Notebook directory:", notebook_dir)
home_dir = notebook_dir
datetime_start_string = datetime.datetime.now().strftime("%Y%m%d%H%M")

#create folder for saving the results
results_dir = f'{home_dir}/{datetime_start_string}'
os.makedirs(results_dir, exist_ok=True)

# Load the labeled proposals (assumed to be in the JSON format as described)
proposals = None
proposal_file_path = f'{home_dir}/labeled_proposals_edge_boxes.json'
with open(proposal_file_path, 'r') as f:
    proposals = json.load(f)


# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

file_list = list(proposals.keys())  # Image files in the proposals
img_dir = f'{home_dir}/Potholes/annotated-images'

def get_datasets(size=(128,128), transform_training = None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, target_background_ratio=0.5, batch_size=32):
    if transform_training is None:
        transform_train = T.Compose([
            T.RandomHorizontalFlip(),  # Randomly flip the image vertically
            T.RandomRotation(10),  # Randomly rotate the image by 20 degrees
            T.Resize(size),  # Resize to 128x128
            T.ToTensor(),  # Convert to tensor
        ])
    else:
        transform_train = transform_training


    transform = T.Compose([
        T.Resize(size),  # Resize to 128x128
        T.ToTensor(),  # Convert to tensor
    ])

    # Split the training data into training and validation sets
    train_files, test_files = train_test_split(file_list, test_size=test_ratio + val_ratio, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)

    target_ratio = 1 - target_background_ratio

    print(f"Number of training images: {len(train_files)}")
    print(f"Number of validation images: {len(val_files)}")
    print(f"Number of test images: {len(test_files)}")

    # Define the datasets using the split files
    train_dataset = PotholeDatasetFromProposals(
        file_list=train_files,
        img_dir=img_dir,
        proposals=proposals,
        transform=transform_train,
        target_ratio=target_ratio,  # 30% potholes, 70% background
        apply_ratio=True
    )

    val_dataset = PotholeDatasetFromProposals(
        file_list=val_files,
        img_dir=img_dir,
        proposals=proposals,
        transform=transform,
        target_ratio=target_ratio,  # 30% potholes, 70% background
        apply_ratio=True
    )

    test_dataset = PotholeDatasetFromProposals(
        file_list=test_files,
        img_dir=img_dir,
        proposals=proposals,
        transform=transform,
        target_ratio=target_ratio,  # 30% potholes, 70% background
        apply_ratio=False,
        test=True
    )

    #calculate class imbalance
    def calculate_class_imbalance(dataset):
        samples = dataset.samples
        background_count = sum(1 for sample in samples if sample["bbox"]["label"] == "background")
        pothole_count = sum(1 for sample in samples if sample["bbox"]["label"] == "pothole")

        print(f"Number of background samples: {background_count}")
        print(f"Number of pothole samples: {pothole_count}")
        
        total_samples = len(samples)
        print(f"Ratio of potholes: {pothole_count / total_samples:.2f}")
        print(f"Ratio of background: {background_count / total_samples:.2f}")

    print("Training set:")
    calculate_class_imbalance(train_dataset)

    print("\nValidation set:")
    calculate_class_imbalance(val_dataset)

    print("\nTest set:")
    calculate_class_imbalance(test_dataset)

    
    # Data loaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_files, val_files, test_files


train_loader, val_loader, test_loader, train_files, val_files, test_files = get_datasets(size=(128,128), batch_size=32)

# %%
import xml.etree.ElementTree as ET
import os
import cv2
import matplotlib.pyplot as plt

def load_ground_truth(img_dir, file_list):
    """
    Load ground truth bounding boxes from Pascal VOC XML files.
    
    Args:
        img_dir (str): Directory containing the images and their annotation files.
        file_list (list): List of image file names.
    
    Returns:
        dict: A dictionary with image file names as keys and ground truth bounding boxes as values.
    """
    ground_truth = {}
    
    for img_name in file_list:
        xml_file = os.path.join(img_dir, img_name.replace('.jpg', '.xml'))
        boxes = []
        
        if not os.path.exists(xml_file):
            print(f"Annotation file missing for {img_name}")
            continue
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            boxes.append({
                "x": xmin,
                "y": ymin,
                "width": xmax - xmin,
                "height": ymax - ymin,
                "label": label
            })
        
        ground_truth[img_name] = boxes
    
    return ground_truth

def plot_images_with_gt_and_bboxes(images, proposals, img_dir, ground_truth, denormalize=False):
    """
    Plot full images with bounding boxes for pothole proposals and ground truth.
    
    Args:
        images (list): List of image file names.
        proposals (dict): Dictionary containing bounding box proposals for each image.
        img_dir (str): Directory where images are stored.
        ground_truth (dict): Dictionary containing ground truth bounding boxes for each image.
        denormalize (bool): Whether to reverse normalization applied to the images.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    
    for i, img_name in enumerate(images):
        # Load the original image
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get proposals for the image
        img_proposals = proposals.get(img_name, [])
        
        # Draw bounding boxes for pothole proposals
        for proposal in img_proposals:
            if proposal['label'] == 'pothole':
                x_min = proposal['x']
                y_min = proposal['y']
                x_max = x_min + proposal['width']
                y_max = y_min + proposal['height']
                
                # Draw the bounding box
                color = (0, 255, 0)  # Green for proposals
                thickness = 2
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Draw ground truth bounding boxes
        img_gt = ground_truth.get(img_name, [])
        for gt_box in img_gt:
            x_min = gt_box['x']
            y_min = gt_box['y']
            x_max = x_min + gt_box['width']
            y_max = y_min + gt_box['height']
            
            # Draw the ground truth bounding box
            color = (255, 0, 0)  # Red for ground truth
            thickness = 2
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Plot the image with bounding boxes
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(img_name)
    
    plt.tight_layout()
    plt.show()

sample_images_reduced = test_files[7:15]
# Generate the ground truth dictionary
ground_truth_reduced = load_ground_truth(img_dir, sample_images_reduced)

# Adjust the range as needed
plot_images_with_gt_and_bboxes(sample_images_reduced, proposals, img_dir, ground_truth_reduced)


# %%
USE_LOCAL_MODEL = False

# %% [markdown]
# ## Define network

# %%
from torchvision import models
import torch.nn as nn

def get_efficient_net():
    # EfficientNet with pretrained weights
    efficientnet_pretrained = models.efficientnet_b1(pretrained=True)
    # Modify the classifier for binary output
    efficientnet_pretrained.classifier[1] = nn.Linear(efficientnet_pretrained.classifier[1].in_features, 1)

    # EfficientNet without pretrained weights
    efficientnet = models.efficientnet_b1(pretrained=False)
    # Modify the classifier for binary output
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 1)
    return efficientnet_pretrained, efficientnet

# %% [markdown]
# ## Train

# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# %%
import torch
torch.cuda.empty_cache()
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# %%
USE_PRETRAINED = True
USE_LOCAL_MODEL = False

# %%
import torch
import torch.optim as optim
import datetime

def train(net, train_loader, val_loader, epochs=5, loss=None, opt=None, output_name="model"):
    model = net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.BCEWithLogitsLoss() if loss is None else loss
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, momentum=0.9) if opt is None else opt

    model = model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, bbox,img_name in train_loader:
            images, labels = images.to(device), labels.float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1)  # Reshape to match labels shape
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_images, val_labels, bbox, img_name in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.float().to(device)

                val_outputs = model(val_images)
                val_outputs = val_outputs.view(-1)  # Reshape to match labels shape
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                predicted = (val_outputs > 0.5).float()
                correct += (predicted == val_labels).sum().item()
                total += val_labels.size(0)

        val_loss = val_loss / len(test_loader)
        best_val_loss = min(best_val_loss, val_loss)
        if val_loss == best_val_loss:
            torch.save(model.state_dict(), f'{home_dir}/{datetime_start_string}/{output_name}_best.pth')
        val_acc = correct / total
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    #plot the loss
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{home_dir}/{datetime_start_string}/{output_name}_loss.png')

    # Save the trained model
    torch.save(model.state_dict(), f'{home_dir}/{datetime_start_string}/{output_name}.pth')

    return model

def load_model(model, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(f'{home_dir}/{model_name}'))
    model.eval()
    model.to(device)
    return model


# %%
def calculate_accuracy_per_class(model, test_loader, device, confidence_thresholds=[0.5,0.9,0.1]):
    """
    Calculate test accuracy per class for multiple confidence thresholds.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (str): Computation device ('cpu' or 'cuda').
        confidence_thresholds (list): List of confidence thresholds to evaluate.

    Returns:
        dict: A dictionary with confidence thresholds as keys and per-class accuracy as values.
    """
    model.eval()
    accuracy_results = {}

    with torch.no_grad():
        for confidence in confidence_thresholds:
            correct = {0: 0, 1: 0}
            total = {0: 0, 1: 0}

            for images, labels, bbox, img_name in test_loader:
                images, labels = images.to(device), labels.float().to(device)

                outputs = model(images)
                predicted = (outputs > confidence).float()

                for label, pred in zip(labels.view(-1), predicted.view(-1)):
                    label = int(label)
                    pred = int(pred)
                    correct[label] += int(label == pred)
                    total[label] += 1

            accuracy = {label: correct[label] / total[label] if total[label] > 0 else 0.0 for label in total}
            accuracy_results[confidence] = accuracy

    return accuracy_results

# %% [markdown]
# ## Prediction for potholes
# Out of the proposals made for the test images, we predict if they are potholes using our neural network (model). Then this predictions are saved in a variable called *model_predictions_score*. The closest it is to 1, the most likeley the proposal is a 0. 
# 
# We use a **threshold** *iou_threshold* to determine if a proposal is or isn't a pothole.

# %%
def generate_model_predictions(dataloader, model, device):
    """
    Generate model predictions for test images using a DataLoader.

    Args:
        dataloader (DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): Trained model for pothole detection.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        dict: A dictionary where keys are image filenames and values are lists of tuples:
              (predicted_score, true_label, bbox) for each proposal.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = {}

    with torch.no_grad():
        for images, labels, bboxes, img_names in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs = model(images)
            outputs = torch.sigmoid(outputs).squeeze()  # Apply sigmoid activation
            
            # Process batch predictions
            for i in range(len(outputs)):
                img_name = img_names[i]
                pred_score = outputs[i].item()
                true_label = labels[i].item()
                bbox = bboxes[i].tolist()

                # Append prediction to the corresponding image
                if img_name not in predictions:
                    predictions[img_name] = []
                predictions[img_name].append((pred_score, true_label, bbox))
    
    return predictions

# %%
def plot_images_with_model_predictions(images, proposals, img_dir, ground_truth, model_predictions_score, confidence=0.5, showFalsePositives=True, model_name = "model",desc = ""):
    """
    Plot test images with bounding boxes for ground truth and predicted pothole proposals.
    
    Args:
        images (list): List of image file names.
        proposals (dict): Dictionary containing bounding box proposals for each image.
        img_dir (str): Directory where images are stored.
        ground_truth (dict): Dictionary containing ground truth bounding boxes for each image.
        model_predictions_score (dict): Dictionary of score predictions from the model with 
    """
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    
    for i, img_name in enumerate(images):
        # Load the original image
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get proposals for the image
        img_proposals = proposals.get(img_name, [])
        img_predictions = model_predictions_score.get(img_name, [])
        
        # Draw bounding boxes for predicted potholes)
        for proposal, output in zip(img_proposals, img_predictions):
            prediction, label, bbox = output
            if prediction > confidence:
                x_min = proposal['x']
                y_min = proposal['y']
                x_max = x_min + proposal['width']
                y_max = y_min + proposal['height']

                if showFalsePositives:
                    # Draw the bounding box
                    color = (255, 0, 255)  # Purple for false positives
                    if label == 1:
                        color = (0, 255, 0)
                else:
                    color = (0, 255, 0)

                thickness = 2
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Draw ground truth bounding boxes
        img_gt = ground_truth.get(img_name, [])
        for gt_box in img_gt:
            x_min = gt_box['x']
            y_min = gt_box['y']
            x_max = x_min + gt_box['width']
            y_max = y_min + gt_box['height']
            
            # Draw the ground truth bounding box
            color = (0, 0, 255)  # Red for ground truth
            thickness = 2
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Plot the image with bounding boxes
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(img_name)
    
    plt.tight_layout()
    plt.savefig(f'{home_dir}/{datetime_start_string}/{model_name}_predictions_{desc}_{confidence}.png')
    #save

# %%
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    Args:
        box1 (dict): Bounding box in {'x': xmin, 'y': ymin, 'width': w, 'height': h}.
        box2 (dict): Bounding box in {'x': xmin, 'y': ymin, 'width': w, 'height': h}.
    Returns:
        float: IoU value.
    """
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = box1['width'] * box1['height']
    area_box2 = box2['width'] * box2['height']
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else 0


# %%
import numpy as np
def get_AP(images, proposals, ground_truth, model_predictions, iou_threshold=0.5):
    """
    Calculate and plot the Precision-Recall (PR) curve for all images combined.
    
    Args:
        images (list): List of image file names. We are choosing the test images
        proposals (dict): Dictionary containing bounding box proposals for each image.
        ground_truth (dict): Dictionary of ground truth bounding boxes for each image.
        model_predictions (dict): Dictionary of model predictions (confidence scores) for each proposal.
        iou_threshold (float): IoU threshold to consider a prediction as true positive.
    
    Returns:
        float: Average Precision (AP) for all images combined.
    """
    # Combine proposals, predictions, and image associations across all images
    combined_data = []
    for img_name in images:
        img_proposals = proposals.get(img_name, [])
        img_predictions = model_predictions.get(img_name, [])
        combined_data.extend([(img_name, proposal, pred) for proposal, pred in zip(img_proposals, img_predictions)])

    # Sort combined data by prediction confidence (descending)
    combined_data.sort(key=lambda x: x[2], reverse=True)

    tp = []  # True positives
    fp = []  # False positives
    matched_gt = {img_name: set() for img_name in images}  # Track matched ground truth boxes per image
    precisions = []
    recalls = []

    total_gt_boxes = sum([len(ground_truth.get(img_name, [])) for img_name in images])

    # Process each proposal in descending confidence order
    for img_name, proposal, pred_score in combined_data:
        best_iou = 0
        matched_gt_idx = -1

        # Check IoU with ground truth boxes for the corresponding image
        img_gt_boxes = ground_truth.get(img_name, [])
        for gt_idx, gt_box in enumerate(img_gt_boxes):
            if gt_idx not in matched_gt[img_name]:
                iou = calculate_iou(proposal, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_idx = gt_idx

        # Determine TP or FP
        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched_gt[img_name].add(matched_gt_idx)  # Mark the ground truth box as matched
        else:
            tp.append(0)
            fp.append(1)

        # Cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Precision and Recall at this point
        precision = tp_cumsum[-1] / (tp_cumsum[-1] + fp_cumsum[-1]) if (tp_cumsum[-1] + fp_cumsum[-1]) > 0 else 0
        recall = tp_cumsum[-1] / total_gt_boxes if total_gt_boxes > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Append final point to ensure curve ends at Precision = 0, Recall = 1
    #precisions.append(0.0)
    #recalls.append(1.0)

    # Calculate Average Precision (AP) as area under the Precision-Recall curve
    ap = np.trapz(precisions, recalls) if len(recalls) > 0 else 0.0

    return ap, precisions, recalls

# %%
# Generate the ground truth dictionary
def plot_AP_before_NMS(model_name, test_files, proposals, img_dir, predictions, iou_threshold=0.3):
    sample_images = test_files
    ground_truth = load_ground_truth(img_dir, sample_images)
    # Calculate and plot PR curve for all images
    ap, precision, recall = get_AP(
        sample_images, proposals, ground_truth, predictions, iou_threshold
    )

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f"IoU Threshold = {iou_threshold}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve before NMS")
    plt.legend()
    plt.grid()
    plt.savefig(f'{home_dir}/{datetime_start_string}/{model_name}_before_NMS.png')
    plt.show()
    #save
    print(f"AP before NMS: {ap:.4f}")


    return ap, precision, recall

# %% [markdown]
# **Recall did not reach 0 because not all ground truth boxes were matched by the proposals (i.e., some ground truth instances were missed, resulting in false negatives).**
# 
# We visualise some of the proposals in the order they are arranged, determined by their score (*model_score_proposals*), which is the possibility with which they are classified as proposals according to the model.

# %%
def visualize_top_proposals(images, img_dir, proposals, ground_truth, model_predictions, iou_threshold=0.5, display_from_pos=0, num_images_display=10):
    """
    Visualize the top N proposals with their corresponding images, predicted bounding boxes,
    and matching ground truth if applicable. Draw unmatched ground truths in orange.
    
    Args:
        images (list): List of image file names.
        proposals (dict): Dictionary containing bounding box proposals for each image.
        ground_truth (dict): Dictionary of ground truth bounding boxes for each image.
        model_predictions (dict): Dictionary of model predictions (confidence scores) for each proposal.
        iou_threshold (float): IoU threshold to consider a prediction as true positive.
        display_from_pos (int): Proposal index to start visualization from (ordered by confidence score).
        num_images_display (int): Number of proposals to visualize.
    """
    # Generate the ground truth dictionary
    ground_truth = load_ground_truth(img_dir, images)

    # Combine proposals, predictions, and image associations across all images
    combined_data = []
    for img_name in images:
        img_proposals = proposals.get(img_name, [])
        img_predictions = model_predictions.get(img_name, [])
        combined_data.extend([(img_name, proposal, pred) for proposal, (pred, label, bbox) in zip(img_proposals, img_predictions)])

    # Sort combined data by prediction confidence (descending)
    combined_data.sort(key=lambda x: x[2], reverse=True)

    # Take the top N proposals
    top_proposals = combined_data[display_from_pos:display_from_pos + num_images_display]

    # Track matched ground truth boxes
    matched_ground_truths = {img_name: [] for img_name in images}

    # Visualize each of the top N proposals
    for idx, (img_name, proposal, pred_score) in enumerate(top_proposals):
        # Load the corresponding image
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw the proposal on the image
        x_min, y_min = proposal['x'], proposal['y']
        x_max = x_min + proposal['width']
        y_max = y_min + proposal['height']
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for proposal

        # Check for matching ground truth
        img_gt_boxes = ground_truth.get(img_name, [])
        matched_gt = None
        for gt_box in img_gt_boxes:
            if gt_box not in matched_ground_truths[img_name]:  # Only consider unmatched ground truths
                iou = calculate_iou(proposal, gt_box)
                if iou >= iou_threshold:
                    matched_gt = gt_box
                    matched_ground_truths[img_name].append(gt_box)  # Mark as matched
                    break

        # If there's a matching ground truth, draw it in red
        if matched_gt:
            x_min_gt, y_min_gt = matched_gt['x'], matched_gt['y']
            x_max_gt = x_min_gt + matched_gt['width']
            y_max_gt = y_min_gt + matched_gt['height']
            cv2.rectangle(image, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (255, 0, 0), 2)  # Red for matching ground truth
        else:
            # Draw all unmatched ground truths in orange
            for gt_box in img_gt_boxes:
                if gt_box not in matched_ground_truths[img_name]:  # Only draw unmatched ground truths
                    x_min_gt, y_min_gt = gt_box['x'], gt_box['y']
                    x_max_gt = x_min_gt + gt_box['width']
                    y_max_gt = y_min_gt + gt_box['height']
                    cv2.rectangle(image, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (255, 165, 0), 2)  # Orange for unmatched ground truths

        # Plot the image
        plt.figure(figsize=(2, 2))
        plt.imshow(image)
        plt.title(f"Proposal {idx + 1}: Confidence = {pred_score:.2f}\nImage: {img_name}\n{'Matched GT' if matched_gt else 'No Match'}")
        plt.axis('off')
        plt.show()


# %% [markdown]
# We can see how for proposal 8 it is a match but because the ground truth has been removed for obtaining AP, it says it is not a match. This should be corrected with NMS:
# 
# ## NMS

# %%
def apply_nms(images, proposals, model_predictions_score, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to the proposals of each image.

    Args:
        proposals (dict): Dictionary of bounding box proposals for each image.
        model_predictions_score (dict): Dictionary of model prediction scores for each proposal.
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        dict: Updated proposals after NMS for each image.
        dict: Updated prediction scores after NMS for each image.
    """
    updated_proposals = {}
    updated_scores = {}
    for img_name in images:
        img_proposals = proposals[img_name]
        scores = [pred_score for pred_score, label, bb in model_predictions_score[img_name]]
        labels = [label for pred_score, label, bb in model_predictions_score[img_name]]

        if len(img_proposals) == 0:
            updated_proposals[img_name] = []
            updated_scores[img_name] = []
            continue

        # Convert proposals to the format required by NMS: [x1, y1, x2, y2]
        boxes = torch.tensor([
            [proposal['x'], proposal['y'], proposal['x'] + proposal['width'], proposal['y'] + proposal['height']]
            for proposal in img_proposals
        ], dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)

        # Apply NMS
        keep_indices = nms(boxes, scores, iou_threshold)

        # Retain proposals and scores after NMS
        updated_proposals[img_name] = [img_proposals[i] for i in keep_indices]
        updated_scores[img_name] = [model_predictions_score[img_name][i] for i in keep_indices]

    return updated_proposals, updated_scores

# %% [markdown]
# Nice! We see how the previous proposal 8 does not appear anymore, because it has been removed by NMS!
# 
# Also, the AP improved!

# %% [markdown]
# ## Parameters to change:
# - The IoU to consider something a proposal when obtaining AP: **iou_threshold**
# - The IoU to remove a proposal when it overlaps with another one for NMS: **iou_threshold**

# %%
iou_threshold = 0.3
iou_threshold_nms = 0.3

# %%
def visualize_nms(test_files, img_dir, proposals, model_predictions, iou_threshold=0.5, iou_threshold_nms=0.5, model_name="model"):    
    sample_images_reduced = test_files[7:15]
    sample_images = test_files
    # Generate the ground truth dictionary
    ground_truth_reduced = load_ground_truth(img_dir, sample_images_reduced)

    # Without NMS
    plot_images_with_model_predictions(sample_images_reduced, proposals, img_dir, ground_truth_reduced, model_predictions, iou_threshold, model_name=model_name, desc=f"before_NMS_{iou_threshold_nms}")

    # With NMS
    nms_proposals, nms_predictions_score = apply_nms(sample_images, proposals, model_predictions, iou_threshold=iou_threshold_nms)

    plot_images_with_model_predictions(sample_images_reduced, nms_proposals, img_dir, ground_truth_reduced, nms_predictions_score, iou_threshold, model_name=model_name, desc=f"after_NMS_{iou_threshold_nms}")

    ## See the AP
    # Generate the ground truth dictionary
    sample_images = test_files
    ground_truth = load_ground_truth(img_dir, sample_images)
    # Calculate and plot PR curve for all images
    ap_nms, precision_nms, recall_nms = get_AP(
        sample_images, nms_proposals, ground_truth, nms_predictions_score, iou_threshold
    )

    return ap_nms, precision_nms, recall_nms

# %%
def plot_AP_after_NMS(model_name, ap = None, precision = None, recall = None, iou_threshold=None, ablation = []):    
    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    if len(ablation) > 0:
        for i in ablation:
            plt.plot(i["recall"], i["precision"], marker='.', label=f"Confidence AP= {iou_threshold if not None else i['tresh_ap']} IoU NMS = {i['threshold']}/n AP = {i['ap']:.4f}")
    else:
        plt.plot(recall, precision, marker='.', label=f"IoU Threshold for AP = {iou_threshold}\nIoU Threshold for NMS = {iou_threshold_nms}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve after NMS")
    plt.legend()
    plt.grid()
    plt.savefig(f'{home_dir}/{datetime_start_string}/{model_name}.png')
    plt.show()
    #save
    if len(ablation) > 0:
        for i in ablation:
            print(f"AP after NMS with IoU Threshold for AP = {iou_threshold if not None else i['tresh_ap']} and IoU Threshold for NMS = {i['threshold']}: {i['ap']:.4f}")
    else:
        print(f"AP after NMS with IoU Threshold for AP = {iou_threshold} and IoU Threshold for NMS = {iou_threshold_nms}: {ap:.4f}")

# %%
iou_threshold = 0.3
iou_threshold_nms = 0.5

# %%
# get datasets 
train_loader, val_loader, test_loader, train_files, val_files, test_files = get_datasets(size=(128,128), batch_size=32)

# get models
efnt_pre, efnet = get_efficient_net()

models = [{"name": "efficientnet_pretrained", "epochs": 5, "model": efnt_pre},{"name": "efficientnet", "epochs": 20, "model": efnet}]
SKIP_TRAINING = False

for m in models:
    if not SKIP_TRAINING:
        # Train the model
        model = train(m["model"], train_loader, val_loader, epochs=m["epochs"], output_name=m["name"])
    else:
        model = load_model(m["model"], f'{m["name"]}_best.pth')

    # Generate model predictions for test images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Calculating accuracy for model: {m['name']}")
    ##   accuracy_results = calculate_accuracy_per_class(model, test_loader, device)
    ## print(f"Accuracy results for {m['name']}: {accuracy_results}")

    model_predictions_score = generate_model_predictions(test_loader, model, device)
    print(f"Model predictions for {m['name']} generated")

    nms_tresholds = [0.1, 0.2, 0.35, 0.5]
    ap_treshold = [0.1, 0.3, 0.5]
    final_ap = None
    final_ap_threshold = None
    
    ablation = []
    for a in ap_treshold:
        print(f"AP Threshold: {a}")
        ap, precision, recall = plot_AP_before_NMS(m["name"], test_files, proposals, img_dir, model_predictions_score, iou_threshold=a)
        ablation.append({"threshold": None, "ap": ap, "precision": precision, "recall": recall, "tresh_ap": f'{a}'})
        if final_ap is None:
            final_ap = ap
            final_ap_threshold = a
        else:
            if ap > final_ap:
                final_ap = ap
                final_ap_threshold = a

    for t in nms_tresholds:
        print(f"Threshold: {t}")
        ap_nms, precision_nms, recall_nms = visualize_nms(test_files, img_dir, proposals, model_predictions_score, iou_threshold=final_ap_threshold, iou_threshold_nms=t, model_name=m["name"])
        ablation.append({"threshold": t, "ap": ap_nms, "precision": precision_nms, "recall": recall_nms})

    plot_AP_after_NMS(m["name"], iou_threshold=final_ap_threshold, ablation = ablation)


