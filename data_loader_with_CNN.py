# Cell
import random
from matplotlib import patches
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os
import xml.etree.ElementTree as ET

class PotholeDatasetFromProposals(Dataset):
    def __init__(self, file_list, img_dir, proposals, transform=None, target_ratio=7.0, apply_ratio=False):
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
        

    def __getitem__(self, idx):
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
        
        return cropped_image, label

# Cell
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces to 64x64
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces to 32x32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # Reduces to 16x16
        
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust the in_features based on image size after convolutions
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Cell
import json

home_dir = '.' # Update with the actual path

# Load the labeled proposals (assumed to be in the JSON format as described)
proposals = None
proposal_file_path = f'{home_dir}/labeled_proposals_edge_boxes.json'
with open(proposal_file_path, 'r') as f:
    proposals = json.load(f)


# Cell
from sklearn.model_selection import train_test_split

file_list = list(proposals.keys())  # Image files in the proposals
img_dir = f'{home_dir}/Potholes/annotated-images' 

transform= T.Compose([
    T.Resize((128, 128)),  # Resize to 128x128
    T.ToTensor(),  # Convert to tensor
])

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split the training data into training and validation sets
train_files, test_files = train_test_split(file_list, test_size=test_ratio + val_ratio, random_state=42)
val_files, test_files = train_test_split(test_files, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)

print(f"Number of training images: {len(train_files)}")
print(f"Number of validation images: {len(val_files)}")
print(f"Number of test images: {len(test_files)}")

# Define the datasets using the split files
train_dataset = PotholeDatasetFromProposals(
    file_list=train_files,
    img_dir=img_dir,
    proposals=proposals,
    transform=transform,
    target_ratio=0.5,  # 30% potholes, 70% background
    apply_ratio=True
)

val_dataset = PotholeDatasetFromProposals(
    file_list=val_files,
    img_dir=img_dir,
    proposals=proposals,
    transform=transform,
    target_ratio=0.5,  # 30% potholes, 70% background
    apply_ratio=True
)

test_dataset = PotholeDatasetFromProposals(
    file_list=test_files,
    img_dir=img_dir,
    proposals=proposals,
    transform=transform,
    target_ratio=0.5,  # 30% potholes, 70% background
    apply_ratio=True
)


# Cell
from torch.utils.data import DataLoader

# Data loaders for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cell
# Display the first batch of images
images, targets = next(iter(train_loader))
#plot the images in the batch, along with the corresponding labels
import matplotlib.pyplot as plt
import numpy as np

#plot background and plot hole images
fig, axs = plt.subplots(1, 8, figsize=(16, 8))
for i, ax in enumerate(axs.flat):
    ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title("Pothole" if targets[i] == 1 else "Background")
plt.show()


# Cell
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


# Cell
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, f1, f3_in, f3_out, f3dbl_in, f3dbl_out, fpool_out):
        super(InceptionModule, self).__init__()

        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=1),
            nn.BatchNorm2d(f1,momentum=0.9),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, f3_in, kernel_size=1),
            nn.Conv2d(f3_in, f3_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3_out,momentum=0.9),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv -> 3x3 conv (replacing the 5x5 conv)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, f3dbl_in, kernel_size=1),
            nn.Conv2d(f3dbl_in, f3dbl_out, kernel_size=3, padding=1),
            nn.Conv2d(f3dbl_out, f3dbl_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3dbl_out,momentum=0.9),
            nn.ReLU(inplace=True)
        )

        # 3x3 maxpool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, fpool_out, kernel_size=1),
            nn.BatchNorm2d(fpool_out,momentum=0.9),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        # Concatenate along the channel axis
        outputs = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        return outputs


# Define the full Inception network for binary classification
class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        # Initial convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        # Fully connected layers for binary classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(480, 1)  # Output is a single value for binary classification (sigmoid)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.5)
        x = torch.sigmoid(self.fc(x))  # Sigmoid for binary classification
        return x

# Instantiate the model, loss function, and optimizer
model = InceptionNet()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.RMSprop(model.parameters(), lr=0.0001, momentum=0.9)

# Training loop
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
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
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device), val_labels.float().to(device)

            val_outputs = model(val_images)
            val_outputs = val_outputs.view(-1)  # Reshape to match labels shape
            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()

            predicted = (val_outputs > 0.5).float()
            correct += (predicted == val_labels).sum().item()
            total += val_labels.size(0)

    val_loss = val_loss / len(test_loader)
    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'inception_net.pth')  # Update with the desired path


# Cell
# Load the trained model
model = InceptionNet()
model.load_state_dict(torch.load('inception_net.pth'))  # Update with the
model.eval()
model.to(device)

# Cell
#plot a few predictions
import matplotlib.pyplot as plt
import numpy as np
import os

# Get a batch of test images
images, labels = next(iter(test_loader))

# Make predictions
with torch.no_grad():    
    images, labels = images.to(device), labels.float().to(device)
    outputs = model(images).to(device)
    predicted = (outputs > 0.5).float()

# Directory to save the images
save_dir = 'saved_images'
os.makedirs(save_dir, exist_ok=True)

# Save the images in the batch, along with the corresponding labels and ground truth
for i in range(len(images)):
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title(f"Pred: {'Pothole' if predicted[i] == 1 else 'Background'}\nLabel: {'Pothole' if labels[i] == 1 else 'Background'}")
    
    # Save the figure
    save_path = os.path.join(save_dir, f'image_{i}.png')
    plt.savefig(save_path)
    plt.close(fig)

# Cell
# Calculate the overall accuracy on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images)
        
        # Ensure outputs are in the correct shape and type
        outputs = outputs.squeeze()  # Remove any extra dimensions
        predicted = (outputs > 0.5).float()
        
        # Flatten predicted and labels to 1D tensors
        predicted = predicted.view(-1)
        labels = labels.view(-1)
        
        total += labels.size(0)
        correct_batch = (predicted == labels).sum().item()
        correct += correct_batch
    
accuracy = correct / total
print(f"Number of correct predictions: {correct}/{total}")
print(f"Accuracy: {accuracy}")

def calculate_iou(box1, box2):
    """
    Calculate intersection over union (IoU) between two boxes.
    """
    # Convert to x1, y1, x2, y2 format
    box1_x1, box1_y1 = box1['x'], box1['y']
    box1_x2, box1_y2 = box1['x'] + box1['width'], box1['y'] + box1['height']
    
    box2_x1, box2_y1 = box2['x'], box2['y']
    box2_x2, box2_y2 = box2['x'] + box2['width'], box2['y'] + box2['height']
    
    # Calculate intersection coordinates
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

def apply_nms(boxes, scores, iou_threshold=0.2, score_threshold=0.6):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    """
    # Convert scores to list if they're tensors
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy().flatten()
            
    # Filter boxes based on score threshold
    valid_indices = [i for i, score in enumerate(scores) if score > score_threshold]
    if not valid_indices:
        return []
    
    # Sort boxes by score in descending order
    valid_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
    
    keep = []
    while valid_indices:
        current = valid_indices[0]
        keep.append(current)
        
        # Remove current index
        valid_indices = valid_indices[1:]
        
        # Check remaining boxes
        remaining_indices = []
        for idx in valid_indices:
            iou = calculate_iou(boxes[current], boxes[idx])
            if iou <= iou_threshold:
                remaining_indices.append(idx)
        
        valid_indices = remaining_indices
    
    return keep

def process_image_detections(image_proposals, scores, iou_threshold=0.2, score_threshold=0.5):
    """
    Process all detections for a single image using NMS.
    """
    
    scores = [s[0] if isinstance(s, list) else s for s in scores]
    
    if not isinstance(scores, list):
        scores = scores.cpu().detach().numpy().tolist()

    
    if len(scores) == 0 or len(image_proposals) == 0:
        return [], []
        
    # Get pothole scores and confidence values
    pothole_indices = []
    pothole_scores = []
        
    for i, (prop, score) in enumerate(zip(image_proposals, scores)):
        if score > score_threshold:
            pothole_indices.append(i)
            pothole_scores.append(score)
    
    if not pothole_indices:
        return [], []
    
    # Get pothole boxes and their scores
    pothole_boxes = [image_proposals[i] for i in pothole_indices]
    pothole_scores = torch.tensor(pothole_scores)
    
    # Apply NMS
    keep_indices = apply_nms(pothole_boxes, pothole_scores, iou_threshold, score_threshold)
    
    # Return kept proposals and their scores
    kept_proposals = [pothole_boxes[i] for i in keep_indices]
    kept_scores = [pothole_scores[i].item() for i in keep_indices]
    
    return kept_proposals, kept_scores

def process_predictions(image_name, proposals, model, device, transform=None):
    """
    Process predictions for a single image.
    """
    if transform is None:
        transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
    
    # Get all proposals for the image
    image_proposals = proposals[image_name]
    
    # Create batch of proposals
    proposal_images = []
    valid_proposals = []
    
    # Load and process each proposal
    image_path = os.path.join(img_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    
    for proposal in image_proposals:
        x, y = proposal['x'], proposal['y']
        width, height = proposal['width'], proposal['height']
        
        # Crop and transform proposal
        cropped = image.crop((x, y, x + width, y + height))
        transformed = transform(cropped)
        proposal_images.append(transformed)
        valid_proposals.append(proposal)
    
    if not proposal_images:
        return [], []
    
    # Create batch tensor
    batch = torch.stack(proposal_images).to(device)
    
    # Get predictions
    with torch.no_grad():
        scores = model(batch)
    
    # Apply NMS
    filtered_proposals, kept_scores = process_image_detections(
        valid_proposals, 
        scores, 
        iou_threshold=0.2, 
        score_threshold=0.5
    )
    
    return filtered_proposals, kept_scores

def visualize_detections(image_name, filtered_proposals, kept_scores, save_dir="output"):
    """
    Visualize detection results with bounding boxes and confidence scores.
    """
    # Load image
    image_path = os.path.join(img_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Add bounding boxes and scores
    for proposal, score in zip(filtered_proposals, kept_scores):
        x, y = proposal['x'], proposal['y']
        width, height = proposal['width'], proposal['height']
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)
        
        # Add confidence score
        plt.text(
            x, y-5, f'{score:.2f}',
            color='red', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    plt.axis('off')
    plt.title(f'Detections for {image_name}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{image_name}_detections.png")
    plt.savefig(save_path)
    plt.close()

# Test the NMS pipeline on multiple images
def test_nms_pipeline(num_images=4):
    """
    Test the NMS pipeline on a few test images.
    """
    model.eval()  # Set model to evaluation mode
    
    # Process a few test images
    test_subset = test_files[:num_images]
    
    # Create a figure with subplots
    plt.figure(figsize=(20, 5 * ((num_images + 3) // 4)))
    
    for idx, image_name in enumerate(test_subset, 1):
        # Get predictions and apply NMS
        filtered_proposals, kept_scores = process_predictions(
            image_name, 
            proposals, 
            model, 
            device
        )
        
        # Create subplot
        plt.subplot(((num_images + 3) // 4), 4, idx)
        
        # Load and show image
        image_path = os.path.join(img_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        plt.imshow(image)
        
        # Add bounding boxes and scores
        for proposal, score in zip(filtered_proposals, kept_scores):
            x, y = proposal['x'], proposal['y']
            width, height = proposal['width'], proposal['height']
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # Add confidence score
            plt.text(
                x, y-5, f'{score:.2f}',
                color='red', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        plt.axis('off')
        plt.title(f'Image {idx}: {image_name}')
    
    plt.tight_layout()
    plt.savefig(f"output/{idx}")

# Run the test with 8 images
print("Testing NMS pipeline with example images...")
test_nms_pipeline(8)

# Process all test images if needed
def process_all_test_images():
    """
    Process all test images and save results to a dictionary.
    """
    results = {}
    model.eval()

    for image_name in test_files:
        filtered_proposals, kept_scores = process_predictions(
            image_name, 
            proposals, 
            model, 
            device
        )
        results[image_name] = {
            'proposals': filtered_proposals,
            'scores': kept_scores
        }
        
        # Print progress
        print(f"Processed {image_name}: found {len(filtered_proposals)} detections")
        visualize_detections(image_name, filtered_proposals, kept_scores)
    
    return results

# Uncomment the following line to process all test images:
all_results = process_all_test_images()
