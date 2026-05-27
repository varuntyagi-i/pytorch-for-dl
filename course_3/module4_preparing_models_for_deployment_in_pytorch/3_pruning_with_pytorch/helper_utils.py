import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.models as tv_models
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score


# Define the global random seed value
RANDOM_SEED = 42

# Set seed for PyTorch CPU operations
torch.manual_seed(RANDOM_SEED)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Set seed for PyTorch GPU operations on all available GPUs
    torch.cuda.manual_seed_all(RANDOM_SEED)


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


def show_weights(model, layer_names):
    """
    Visualizes a sample of weights for specified model layers.

    Args:
        model: The neural network model containing the layers.
        layer_names: A string or list of strings representing the layer attributes to inspect.
    """
    # Convert single layer name to a list for uniform processing
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    # Process each requested layer
    for layer_name in layer_names:
        try:
            # Retrieve the layer attribute from the model
            layer = getattr(model, layer_name)

            # Ensure the layer contains a weight parameter
            if not hasattr(layer, 'weight'):
                print(f"Layer '{layer_name}' has no 'weight' attribute.")
                continue

            print(f"--- Weights for '{layer_name}' ---")
            # Detach the weights and convert to a NumPy array
            weights = layer.weight.detach().cpu().numpy()

            # Handle visualization for convolutional layers
            if isinstance(layer, nn.Conv2d):
                if weights.shape[0] >= 2 and weights.shape[1] >= 2:
                    # Concatenate four 3x3 kernels into a 6x6 grid
                    kernel_0_0 = weights[0, 0]
                    kernel_0_1 = weights[0, 1]
                    kernel_1_0 = weights[1, 0]
                    kernel_1_1 = weights[1, 1]
                    grid_6x6 = np.block([[kernel_0_0, kernel_0_1], [kernel_1_0, kernel_1_1]])
                    print(grid_6x6)
                else:
                    # Fallback for layers with small channel dimensions
                    print(weights[0, 0])

            # Handle visualization for linear layers
            elif isinstance(layer, nn.Linear):
                rows, cols = weights.shape
                # Define a slice with a maximum size of 6x6
                slice_rows = min(6, rows)
                slice_cols = min(6, cols)
                # Output the top-left corner of the weight matrix
                print(weights[:slice_rows, :slice_cols])

            else:
                print(f"Visualization for layer type {type(layer)} is not implemented.")

            print("-" * 50)
            print() 

        except AttributeError:
            # Handle cases where the layer name does not exist in the model
            print(f"Layer '{layer_name}' not found in the model.")


class TransformedDataset(Dataset):
    """
    A dataset wrapper that applies specific transformations to a dataset subset.

    Args:
        subset: The underlying dataset or subset of data.
        transform: The transformation functions to be applied to the data samples.
    """
    
    def __init__(self, subset, transform):
        """
        Initializes the TransformedDataset with a subset and a transformation pipeline.

        Args:
            subset: The collection of data to be transformed.
            transform: The operations to perform on each data item.
        """
        # Assign the data subset to the instance
        self.subset = subset
        # Assign the transformation pipeline to the instance
        self.transform = transform
    
    def __len__(self):
        """
        Retrieves the total number of items in the dataset.

        Returns:
            The integer length of the subset.
        """
        # Return the length of the internal subset
        return len(self.subset)

    def __getitem__(self, idx):
        """
        Retrieves and transforms a specific item from the dataset.

        Args:
            idx: The index of the item to fetch.

        Returns:
            A tuple containing the transformed data and its corresponding label.
        """
        # Fetch the image and label from the subset at the given index
        img, label = self.subset[idx]
        # Apply the transformation to the image before returning
        return self.transform(img), label




def get_dataloaders(dataset_path, transformations=[transform_train, transform_val], batch_size=32):
    """
    Prepares data loaders for training and validation by splitting a dataset.

    Args:
        dataset_path: The file path to the root directory of the image dataset.
        transformations: A list containing two transformation pipelines (index 0 for train, 1 for val).
        batch_size: The number of samples to load per batch.
    """
    # Load the source dataset from the specified directory structure
    full_dataset = ImageFolder(root=dataset_path)

    # Define the proportion of data to be used for training
    train_ratio = 0.8
    # Determine the count of training samples based on the ratio
    train_size = int(train_ratio * len(full_dataset))
    # Determine the count of validation samples from the remainder
    val_size = len(full_dataset) - train_size
    # Randomly partition the dataset into two distinct subsets
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Output the calculated sizes for each data subset
    print(f"Train samples (80%):\t\t{len(train_subset)}")
    print(f"Validation samples (20%):\t{len(val_subset)}\n")

    # Wrap the subsets with their respective transformation logic
    train_dataset = TransformedDataset(train_subset, transformations[0])
    val_dataset = TransformedDataset(val_subset, transformations[1])

    # Initialize the training data loader with shuffling enabled
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # Initialize the validation data loader without shuffling
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Log the successful initialization of the loader objects
    print(f"DataLoaders created with {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # Provide the loader objects for use in a training loop
    return train_loader, val_loader


def load_resnet18(path='./pretrained_resnet18_weights/resnet18-f37072fd.pth'):
    """
    Initializes a ResNet18 model and loads weights from a local file.

    Args:
        path: The file system path to the model weight file.
    """
    # Instantiate the model architecture without downloading weights automatically
    model = tv_models.resnet18(weights=None)
    
    # Define the local source for the model parameters
    weights_path = path
    # Load the state dictionary from the specified file path
    state_dict = torch.load(weights_path)
    
    # Map the loaded parameters into the model architecture
    model.load_state_dict(state_dict)

    # Return the prepared model for inference or training
    return model


def replace_final_layer(model, num_classes):
    """
    Modifies the final fully connected layer of the model to match a new number of classes.

    Args:
        model: The neural network model whose final layer is to be replaced.
        num_classes: The number of output classes for the new classification task.
    """
    # Identify the number of input features entering the final layer
    num_ftrs = model.fc.in_features
    
    # Initialize a new linear layer with the correct output dimensions
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Return the updated model architecture
    return model


def training_loop(model, train_loader, val_loader, num_epochs, device, num_classes):
    """
    Executes a standard training and validation routine for a neural network.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader providing the training dataset.
        val_loader: DataLoader providing the validation dataset.
        num_epochs: Total number of iterations over the entire dataset.
        device: The processing unit (CPU or GPU) to use for computation.
        num_classes: The number of distinct categories in the dataset.
    """
    # Initialize the cross-entropy loss function for classification
    loss_function = nn.CrossEntropyLoss()
    
    # Configure the Adam optimizer
    # Update all parameters with a smaller learning rate suitable for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Set up a scheduler to decrease the learning rate when validation loss stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Initialize evaluation metrics for multiclass classification on the target device
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_score_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    
    # Transfer the model architecture to the specified hardware
    model = model.to(device)

    # Log the commencement of the training session
    print(f"--- Starting Training ---")
    
    # Iterate through the specified number of training cycles
    for epoch in range(num_epochs):
        
        # Enable gradients and specific behaviors for training
        model.train()
        # Track the total loss accumulated during the training phase
        train_loss = 0.0
        # Initialize a visual progress tracker for training batches
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        # Process each batch of training data
        for inputs, labels in train_progress_bar:
            # Move batch tensors to the active computation device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradient buffers from the previous iteration
            optimizer.zero_grad()
            
            # Compute model predictions based on current weights
            outputs = model(inputs)
            # Determine the error between predictions and ground truth
            loss = loss_function(outputs, labels)
            
            # Backpropagate the error to calculate parameter gradients
            loss.backward()
            # Adjust model parameters based on the calculated gradients
            optimizer.step()
            
            # Update the total training loss for this epoch
            train_loss += loss.item()
            # Display the current average loss in the progress bar
            train_progress_bar.set_postfix(loss=f"{(train_loss / (train_progress_bar.n + 1)):.4f}")
            
        # Disable gradient computation and enable evaluation behaviors
        model.eval()
        # Track the total loss accumulated during the validation phase
        val_loss = 0.0
        
        # Initialize a visual progress tracker for validation batches
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        # Ensure no weight updates occur during the validation phase
        with torch.no_grad():
            # Process each batch of validation data
            for inputs, labels in val_progress_bar:
                # Move batch tensors to the active computation device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Generate model predictions for validation samples
                outputs = model(inputs)
                # Compute the validation error
                loss = loss_function(outputs, labels)
                
                # Update the total validation loss for this epoch
                val_loss += loss.item()
                # Determine the class with the highest probability
                _, predicted = torch.max(outputs.data, 1)
                
                # Accumulate statistics for each performance metric
                accuracy_metric.update(predicted, labels)
                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                f1_score_metric.update(predicted, labels)
                
                # Display the current average validation loss in the progress bar
                val_progress_bar.set_postfix(loss=f"{(val_loss / (val_progress_bar.n + 1)):.4f}")
                
        # Calculate the mean loss for both phases over the entire epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Finalize the metric calculations for the current epoch
        accuracy = accuracy_metric.compute().item() * 100
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1_score = f1_score_metric.compute().item()
        
        # Clear metric states for the next epoch's calculations
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()
        
        # Update the learning rate if the validation loss has plateaued
        scheduler.step(avg_val_loss)
        
        # Display a summary of performance and loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Log the completion of the training process
    print("\n--- Finished Training ---")
    
    # Consolidate the final performance statistics into a single structure
    final_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    # Return the optimized model and its final validation metrics
    return model, final_metrics


def save_unpruned_model_and_metrics(
    model, 
    metrics, 
    state_dict_filename="unpruned_model_state_dict.pth", 
    metrics_filename="unpruned_metrics.pkl"
):
    """
    Persists the model state dictionary and evaluation metrics to local storage.

    Args:
        model: The trained neural network model to be saved.
        metrics: A dictionary containing performance evaluation data.
        state_dict_filename: The target filename for the saved model weights.
        metrics_filename: The target filename for the serialized metrics data.
    """
    # Save the model weights and parameters to a file
    torch.save(model.state_dict(), state_dict_filename)
    print(f"Unpruned model state dictionary saved to {state_dict_filename}")

    # Open a file stream to serialize the metrics dictionary
    with open(metrics_filename, 'wb') as f:
        # Write the metrics data using pickle serialization
        pickle.dump(metrics, f)
    print(f"Unpruned model metrics saved to {metrics_filename}")


def save_pruned_model_and_metrics(
    model, 
    metrics, 
    state_dict_filename="pruned_model_permanent_state_dict.pth", 
    metrics_filename="pruned_metrics.pkl"
):
    """
    Saves the model state dictionary and associated performance metrics to files.

    Args:
        model: The pruned model instance containing the weights to be saved.
        metrics: A dictionary containing the evaluation results and metrics.
        state_dict_filename: The destination path for the model state dictionary file.
        metrics_filename: The destination path for the serialized metrics file.
    """
    # Save the model's weight parameters to the specified disk location
    torch.save(model.state_dict(), state_dict_filename)
    print(f"Permanently pruned model state dictionary saved to {state_dict_filename}")

    # Open the target file in binary write mode for serialization
    with open(metrics_filename, 'wb') as f:
        # Serialize the metrics dictionary using the pickle module
        pickle.dump(metrics, f)
    print(f"Permanently pruned model metrics saved to {metrics_filename}")


def comparison_report(
    unpruned_state_dict_path, 
    unpruned_metrics_path,
    pruned_state_dict_path, 
    pruned_metrics_path,
    num_epochs, 
    device
):
    """
    Compiles and displays a comprehensive comparison between unpruned and pruned models.

    Args:
        unpruned_state_dict_path: Path to the saved weights of the original model.
        unpruned_metrics_path: Path to the serialized metrics of the original model.
        pruned_state_dict_path: Path to the saved weights of the pruned model.
        pruned_metrics_path: Path to the serialized metrics of the pruned model.
        num_epochs: The number of training epochs completed.
        device: The hardware device used for loading model data.
    """
    print(f"\n--- Final Comparison Report After Running Training for {num_epochs} Epoch(s) ---")

    def count_total_parameters_from_state_dict(state_dict):
        """
        Calculates the sum of all elements in the provided state dictionary.

        Args:
            state_dict: The dictionary of model parameters.
        """
        # Sum the number of elements for every tensor in the dictionary
        return sum(p.numel() for p in state_dict.values())

    def count_nonzero_parameters_from_state_dict(state_dict):
        """
        Calculates the sum of all non-zero elements in the provided state dictionary.

        Args:
            state_dict: The dictionary of model parameters.
        """
        # Filter and count only the elements that are not zero
        return sum(torch.count_nonzero(p).item() for p in state_dict.values())

    # Load the state dictionary for the unpruned model onto the target device
    unpruned_state_dict = torch.load(unpruned_state_dict_path, map_location=device)
    # Load the performance metrics for the unpruned model
    with open(unpruned_metrics_path, 'rb') as f:
        unpruned_metrics = pickle.load(f)

    # Compute parameter counts for the unpruned model
    unpruned_total_params = count_total_parameters_from_state_dict(unpruned_state_dict)
    unpruned_nonzero_params = count_nonzero_parameters_from_state_dict(unpruned_state_dict)
    # Retrieve the file size in bytes and convert to megabytes
    unpruned_size_bytes = os.path.getsize(unpruned_state_dict_path)
    unpruned_size_mb = unpruned_size_bytes / (1024 * 1024)

    # Load the state dictionary for the pruned model onto the target device
    pruned_state_dict = torch.load(pruned_state_dict_path, map_location=device)
    # Load the performance metrics for the pruned model
    with open(pruned_metrics_path, 'rb') as f:
        pruned_metrics = pickle.load(f)

    # Compute parameter counts for the pruned model
    pruned_total_params = count_total_parameters_from_state_dict(pruned_state_dict)
    pruned_nonzero_params = count_nonzero_parameters_from_state_dict(pruned_state_dict)
    # Retrieve the file size in bytes and convert to megabytes
    pruned_size_bytes = os.path.getsize(pruned_state_dict_path)
    pruned_size_mb = pruned_size_bytes / (1024 * 1024)

    # Format the collected data into a readable string summary
    report = f"""
--- Unpruned Model ---
Total Parameters:         {unpruned_total_params:,}
Non-Zero Parameters:      {unpruned_nonzero_params:,}
Saved model size:         {unpruned_size_mb:.2f} MB
Final Accuracy:           {unpruned_metrics['accuracy']:.2f}%
Final Precision (Macro):  {unpruned_metrics['precision']:.4f}
Final Recall (Macro):     {unpruned_metrics['recall']:.4f}
Final F1-Score (Macro):   {unpruned_metrics['f1_score']:.4f}

--- Pruned Model ---
Total Parameters:         {pruned_total_params:,}
Non-Zero Parameters:      {pruned_nonzero_params:,} (Effective Parameters: weights retained for computation)
Saved model size:         {pruned_size_mb:.2f} MB
Final Accuracy:           {pruned_metrics['accuracy']:.2f}%
Final Precision (Macro):  {pruned_metrics['precision']:.4f}
Final Recall (Macro):     {pruned_metrics['recall']:.4f}
Final F1-Score (Macro):   {pruned_metrics['f1_score']:.4f}
"""
    # Print the final formatted report to the console
    print(report)