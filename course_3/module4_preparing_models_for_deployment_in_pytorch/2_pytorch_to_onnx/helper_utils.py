import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm.auto import tqdm



transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.630, 0.554, 0.489],
                         std=[0.248, 0.271, 0.319]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.630, 0.554, 0.489],
                         std=[0.248, 0.271, 0.319]),
])


def dataset_images_per_class(dataset_path):
    """
    Counts and prints the number of images for each class in a dataset directory.

    Args:
        dataset_path (str): The file path to the directory containing class subfolders.

    """
    # Define the image file extensions considered valid for counting
    valid_exts = ('.jpeg', '.jpg', '.JPG', '.Jpg')
    
    # Initialize a dictionary to store the count of images for each class
    class_counts = defaultdict(int)
    
    # Iterate through the contents of the provided dataset path in alphabetical order
    for class_name in sorted(os.listdir(dataset_path)):
        # Construct the absolute path for the potential class directory
        class_dir = os.path.join(dataset_path, class_name)
        
        # Verify if the current path points to an actual directory
        if os.path.isdir(class_dir):
            # Generate a list of files that match the allowed image extensions
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_exts)]
            # Assign the total count of discovered images to the class name
            class_counts[class_name] = len(image_files)
    
    # Output the total number of identified classes to the console
    print(f"Total number of classes: {len(class_counts)}\n")
    # Output the header for the individual class breakdown
    print("Number of images per class:\n")
    # Loop through the dictionary to display each class and its respective image count
    for cls, count in class_counts.items():
        print(f"{cls:25}: {count}")


class TransformedDataset(Dataset):
    """
    A custom dataset wrapper that applies specific transformations to a dataset subset.

    Args:
        subset (Dataset): A subset of a larger dataset or a standalone dataset object.
        transform (callable): A function or transform object that takes in an image 
                              and returns a transformed version.
    """
    
    def __init__(self, subset, transform):
        """
        Initializes the dataset with a specific subset and transformation pipeline.

        Args:
            subset (Dataset): The underlying data subset to be used.
            transform (callable): The transformation logic to apply to the data.
        """
        # Assign the provided data subset to the instance
        self.subset = subset
        # Assign the transformation function to the instance
        self.transform = transform

    def __len__(self):
        """
        Calculates the total number of samples in the dataset.

        Returns:
            length (int): The number of items available in the subset.
        """
        # Return the length of the underlying subset
        return len(self.subset)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset and applies the defined transformation.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            transformed_img (Tensor/PIL Image): The data sample after applying transformations.
            label (any): The ground truth label associated with the data sample.
        """
        # Extract the raw image and label from the subset at the given index
        img, label = self.subset[idx]
        
        # Apply the transformation to the image and return it along with the label
        return self.transform(img), label


def get_dataloaders(dataset_path, transformations=[transform_train, transform_val], batch_size=32):
    """
    Creates and returns training and validation DataLoaders from an ImageFolder dataset.

    Args:
        dataset_path (str): The file path to the root directory of the image dataset.
        transformations (list): A list containing two transformation objects, where 
                                index 0 is for training and index 1 is for validation.
        batch_size (int): The number of samples per batch to load.

    Returns:
        train_loader (DataLoader): The DataLoader instance for the training dataset.
        val_loader (DataLoader): The DataLoader instance for the validation dataset.
    """
    # Load the comprehensive image dataset from the provided directory path
    full_dataset = ImageFolder(root=dataset_path)

    # Define the proportion of the dataset to be allocated for training
    train_ratio = 0.8
    # Compute the integer count of samples intended for the training set
    train_size = int(train_ratio * len(full_dataset))
    # Compute the remaining count of samples for the validation set
    val_size = len(full_dataset) - train_size
    # Partition the full dataset into training and validation subsets randomly
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Print the resulting size of the training subset to the console
    print(f"Train samples (80%):\t\t{len(train_subset)}")
    # Print the resulting size of the validation subset to the console
    print(f"Validation samples (20%):\t{len(val_subset)}\n")

    # Wrap the training subset with the corresponding training transformations
    train_dataset = TransformedDataset(train_subset, transformations[0])
    # Wrap the validation subset with the corresponding validation transformations
    val_dataset = TransformedDataset(val_subset, transformations[1])

    # Initialize the training DataLoader with shuffling enabled and specified batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # Initialize the validation DataLoader with shuffling disabled for consistency
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Provide a confirmation message regarding the successful creation of the loaders
    print(f"DataLoaders created with {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # Return the training and validation data loaders as individual objects
    return train_loader, val_loader
    

def show_image_grid(dataloader, class_names):
    """
    Displays a 3x3 grid of images from a dataloader, reversing normalization for visualization.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader containing the image batches.
        class_names (list): A list of strings representing the names of the dataset classes.

    Returns:
        None: This function displays a plot and does not return a value.
    """
    # Extract a single batch of images and their corresponding labels from the provider
    images, labels = next(iter(dataloader))
    
    # Define the specific mean values used during the initial normalization process
    mean = torch.tensor([0.630, 0.554, 0.489])
    # Define the specific standard deviation values used during the initial normalization process
    std = torch.tensor([0.248, 0.271, 0.319])

    # Initialize a figure with a 3 by 3 grid of subplots for image display
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    
    # Iterate through the grid axes to populate each subplot with an image
    for i, ax in enumerate(axes.flatten()):
        # Validate that the current index does not exceed the available images in the batch
        if i < len(images):
            # Select the image tensor at the current index
            image = images[i]
            # Select the label index at the current index
            label = labels[i]
            
            # Reverse the normalization by multiplying by standard deviation and adding the mean
            # Reshaping is performed to align dimensions for broadcasting across the tensor
            unnormalized_image = image * std[:, None, None] + mean[:, None, None]
            
            # Reorder the tensor dimensions from (Channels, Height, Width) to (Height, Width, Channels)
            img_display = unnormalized_image.permute(1, 2, 0).numpy()
            
            # Constrain the pixel values to the standard range of 0 to 1
            img_display = np.clip(img_display, 0, 1)
            
            # Render the processed image onto the current subplot axis
            ax.imshow(img_display)
            # Set the subplot title to the human-readable class name
            ax.set_title(class_names[label])
            # Disable the axis markers for a cleaner visual representation
            ax.axis("off")
    
    # Adjust the layout to prevent overlapping elements in the grid
    plt.tight_layout()
    # Execute the command to render the final plot
    plt.show()


def adapt_model_for_transfer_learning(model, num_classes):
    """
    Modifies a pre-trained model by freezing its weights and replacing the final 
    classification layer to match a new number of output classes.

    Args:
        model (torch.nn.Module): The pre-trained neural network model to be adapted.
        num_classes (int): The number of output classes for the new task.

    Returns:
        model (torch.nn.Module): The modified model with frozen base layers and a 
                                 newly initialized final linear layer.
    """
    # Iterate through all parameters in the model to disable gradient updates
    for param in model.parameters():
        # Set the gradient requirement to False to freeze the layer weights
        param.requires_grad = False
    
    # Retrieve the number of input features entering the existing final fully connected layer
    num_ftrs = model.fc.in_features
    
    # Define a new linear layer with the correct number of outputs and assign it to the model
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Return the updated model architecture
    return model


def training_loop(model, train_loader, val_loader, num_epochs, device):
    """
    Executes the complete training and validation loop for a PyTorch model.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader providing the training dataset batches.
        val_loader (DataLoader): DataLoader providing the validation dataset batches.
        num_epochs (int): The total number of iterations over the entire dataset.
        device (torch.device): The device (CPU or GPU) where the computations will occur.

    Returns:
        model (torch.nn.Module): The trained model after completing all epochs.
    """
    # Define the standard cross-entropy loss function for classification tasks
    loss_function = nn.CrossEntropyLoss()
    
    # Initialize the Adam optimizer specifically for the final fully connected layer parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Set up a scheduler to decrease the learning rate when the validation loss stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Transfer the model architecture and weights to the target computation device
    model = model.to(device)

    # Print a notification to indicate the start of the training cycle
    print(f"--- Starting Training ---")
    
    # Iterate through the dataset for the specified number of training cycles
    for epoch in range(num_epochs):
        
        # Enable gradient tracking and batch normalization updates by setting training mode
        model.train()
        # Track the cumulative loss for the current training phase
        train_loss = 0.0
        # Initialize a progress monitoring bar for the training batches
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        # Process each batch of data within the training loader
        for inputs, labels in train_progress_bar:
            # Transfer the input data and target labels to the active device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradients from the previous iteration to prevent accumulation
            optimizer.zero_grad()
            
            # Compute the model's predictions for the current batch
            outputs = model(inputs)
            # Calculate the discrepancy between the predictions and the actual labels
            loss = loss_function(outputs, labels)
            
            # Compute the gradients for all trainable parameters using backpropagation
            loss.backward()
            # Adjust the model weights based on the calculated gradients
            optimizer.step()
            
            # Accumulate the batch loss value into the total training loss
            train_loss += loss.item()
            # Refresh the progress bar with the current running average loss
            train_progress_bar.set_postfix(loss=f"{(train_loss / (train_progress_bar.n + 1)):.4f}")
            
        # Disable gradient updates and fix normalization statistics for evaluation
        model.eval()
        # Initialize trackers for validation loss and classification performance
        val_loss = 0.0
        correct = 0
        total = 0
        # Initialize a progress monitoring bar for the validation batches
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        # Suspend gradient computation to reduce memory usage and increase speed
        with torch.no_grad():
            # Process each batch of data within the validation loader
            for inputs, labels in val_progress_bar:
                # Transfer the validation data to the active device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Generate predictions for the validation batch
                outputs = model(inputs)
                # Calculate the loss for the validation batch
                loss = loss_function(outputs, labels)
                
                # Accumulate the batch loss into the total validation loss
                val_loss += loss.item()
                # Determine the class with the highest probability for each sample
                _, predicted = torch.max(outputs.data, 1)
                # Count the total number of validation samples processed
                total += labels.size(0)
                # Increment the count of correctly identified samples
                correct += (predicted == labels).sum().item()
                
                # Refresh the progress bar with current validation loss and accuracy
                val_progress_bar.set_postfix(loss=f"{(val_loss / (val_progress_bar.n + 1)):.4f}", acc=f"{(100 * correct / total):.2f}%")
                
        # Compute the average training loss across all batches in the epoch
        avg_train_loss = train_loss / len(train_loader)
        # Compute the average validation loss across all batches in the epoch
        avg_val_loss = val_loss / len(val_loader)
        # Compute the final accuracy percentage for the validation phase
        accuracy = 100 * correct / total
        
        # Update the learning rate scheduler with the epoch's average validation loss
        scheduler.step(avg_val_loss)
        
        # Output a comprehensive summary of performance for the completed epoch
        print(f"Epoch {epoch+1}/{num_epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Print a final message indicating the conclusion of the training process
    print("\n--- Finished Training ---")
    
    # Return the final state of the trained model
    return model


def show_prediction_grid(images, labels, predictions, classes):
    """
    Displays a grid of images along with their ground truth and predicted labels.
    
    Args:
        images (numpy.ndarray): A batch of image tensors or arrays to be displayed.
        labels (list/numpy.ndarray): The actual class indices for the given images.
        predictions (list/numpy.ndarray): The raw model outputs or probabilities for each image.
        classes (list): A list of strings representing the human-readable class names.

    Returns:
        None: This function renders a matplotlib plot and does not return any value.
    """
    # Set the mean values used during the normalization process for reversal
    mean = np.array([0.630, 0.554, 0.489])
    # Set the standard deviation values used during the normalization process for reversal
    std = np.array([0.248, 0.271, 0.319])
    
    # Initialize a figure with a 3 by 3 subplot arrangement for visualization
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    # Adjust the spacing between subplots to ensure labels are legible
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
    # Iterate through the flattened array of subplot axes
    for i, ax in enumerate(axes.flat):
        # Ensure the current index does not exceed the number of images in the batch
        if i < len(images):
            # Change the image dimension order from (C, H, W) to (H, W, C) for matplotlib compatibility
            img = images[i].transpose((1, 2, 0))
            # Reverse the normalization by multiplying by standard deviation and adding the mean
            img = std * img + mean
            # Clamp the pixel values to the valid range of 0 to 1
            img = np.clip(img, 0, 1)
            
            # Render the processed image in the current subplot
            ax.imshow(img)
            
            # Map the true label index to its corresponding class name string
            true_label = classes[labels[i]]
            # Map the predicted label (index with the highest value) to its class name string
            pred_label = classes[np.argmax(predictions[i])]
            
            # Apply a title to the subplot displaying both labels with conditional formatting
            # Titles appear green if the prediction is correct and red if it is incorrect
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}",
                color=("green" if true_label == pred_label else "red")
            )
            
            # Disable horizontal axis tick marks for a cleaner display
            ax.set_xticks([])
            # Disable vertical axis tick marks for a cleaner display
            ax.set_yticks([])
            
    # Display the final generated grid of predictions
    plt.show()

