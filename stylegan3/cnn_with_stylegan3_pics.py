import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#because of a bug with ImageFolder we coudnt get a result but we still worked a lot on it
# Define the CNN model
class Cnn(nn.Module):
    def __init__(self, ngpu):
        super(Cnn, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Load the saved model
model_save_path = "models/cnns/cnn-sg2-1.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = 1
cnn = Cnn(ngpu).to(device)
cnn.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
cnn.eval()

# Define transformations for validation and test sets
transform_validation_test = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the change_labels function
def change_labels(dataset, old_label, new_label):
    for i in range(len(dataset.samples)):
        path, label = dataset.samples[i]
        if label == old_label:
            dataset.samples[i] = (path, new_label)

# Load synthetic and real images into separate ImageFolder Datasets
synthetic_data = ImageFolder(root='data/generated_images', transform=transform_validation_test)
real_data = ImageFolder(root='data/celebahq', transform=transform_validation_test)

# Example: Set labels for synthetic data to 1 and for real data to 0
change_labels(synthetic_data, 0, 1)
change_labels(real_data, 0, 0)

# Print the length of the datasets
print(f"Length of synthetic dataset: {len(synthetic_data)}")
print(f"Length of real dataset: {len(real_data)}")

# Verify label assignment
print(f"Synthetic data class index: {synthetic_data.class_to_idx}")
print(f"Real data class index: {real_data.class_to_idx}")

# Combine the datasets and create a DataLoader
combined_data = ConcatDataset([synthetic_data, real_data])
combined_loader = DataLoader(combined_data, batch_size=32, shuffle=True)

# Define DataLoaders for validation and test sets if needed
# Assuming you have defined `validation_dataloader` and `test_dataloader`

# Function to display the first 64 images from the DataLoader
def show_images(data, num_images=64, unnormalize=True):
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    count = 0
    for images, labels in data:
        for j in range(images.size(0)):
            if count >= num_images:
                break
            image = images[j].permute(1, 2, 0).numpy()
            if unnormalize:
                image = image * std + mean
            image = np.clip(image, 0, 1)
            ax = axes[count // 8, count % 8]
            ax.imshow(image)
            ax.set_title(f"{'Real' if labels[j] == 0 else 'Fake'}")
            ax.axis('off')
            count += 1
        if count >= num_images:
            break
    plt.show()

# Show the first 64 images from each DataLoader
print("Showing images from combined_loader")
show_images(combined_loader)

# Define DataLoaders for validation and test datasets
# Example placeholders, replace with your actual DataLoaders
validation_dataloader = DataLoader(ConcatDataset([synthetic_data, real_data]), batch_size=32, shuffle=True)
test_dataloader = DataLoader(ConcatDataset([synthetic_data, real_data]), batch_size=32, shuffle=True)

print("Showing images from validation_dataloader")
show_images(validation_dataloader)

print("Showing images from test_dataloader")
show_images(test_dataloader)

# Function to evaluate a DataLoader with a progress bar
def evaluate(loader, total_images):
    correct = 0
    total = 0
    progress_bar = tqdm(total=total_images, desc="Processing Images")
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = cnn(images)
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()
            progress_bar.update(labels.size(0))
            
    progress_bar.close()
    accuracy = correct / total
    return accuracy

# Set the number of images for evaluation (adjust as necessary)
total_images = 20000

# Perform the evaluation
accuracy = evaluate(combined_loader, total_images)

# Plot the results
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color=['blue'])
plt.ylabel('Accuracy')
plt.title('CNN Accuracy on Mixed Synthetic and Real Images')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

print(f'Accuracy: {accuracy:.2f}')
