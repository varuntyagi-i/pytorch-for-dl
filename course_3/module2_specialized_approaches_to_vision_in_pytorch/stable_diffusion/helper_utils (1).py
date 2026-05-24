import os
import random
import zipfile

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, display
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.utils import make_grid
from tqdm.auto import tqdm


def plot_samples_from_dataset(dataset_path):
    """
    Visualizes a 2x3 grid of fruit images based on predefined categories and health statuses.

    The grid maps specific fruits (Apple, Mango, Tomato) to columns and 
    health conditions (Healthy, Rotten) to rows. A random sample is selected 
    for each category from the provided dataset path.

    Arguments:
        dataset_path: The directory path containing the categorized image folders.
    """
    # Initialize the figure and subplots grid with specific dimensions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Define the list of fruits corresponding to grid columns
    fruits = ['Apple', 'Mango', 'Tomato']
    # Define the list of health statuses corresponding to grid rows
    statuses = ['Healthy', 'Rotten']

    # Iterate through each fruit category to handle columns
    for col_idx, fruit in enumerate(fruits):
        # Iterate through each health status to handle rows
        for row_idx, status in enumerate(statuses):
            
            # Generate the directory name based on fruit and status combination
            class_name = f"{fruit}_{status}"
            # Construct the full path to the specific class directory
            class_dir = os.path.join(dataset_path, class_name)
            
            # Retrieve the specific subplot axis for the current row and column
            ax = axes[row_idx, col_idx]
            
            # Verify if the constructed directory exists in the dataset path
            if os.path.isdir(class_dir):
                # List all valid image files within the directory
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg'))]
                
                # Check if the list contains any images
                if image_files:
                    # Select a single random image from the available files
                    random_image = random.choice(image_files)
                    # Create the full path to the selected image
                    img_path = os.path.join(class_dir, random_image)
                    
                    # Read the image data
                    img = plt.imread(img_path)
                    # Display the image on the current axis
                    ax.imshow(img)
                    # Set the title for the subplot using the class name
                    ax.set_title(class_name, fontsize=12, fontweight='bold')
                else:
                    # Display a placeholder text if no images are found in the folder
                    ax.text(0.5, 0.5, 'No Images', ha='center', va='center')
            else:
                # Display a placeholder text if the directory does not exist
                ax.text(0.5, 0.5, f'{class_name}\nNot Found', ha='center', va='center')
            
            # Remove axis ticks and labels for cleaner visualization
            ax.axis('off')

    # Adjust subplot parameters for a compact layout
    plt.tight_layout()
    # Render the final figure
    plt.show()


def load_model(model_path, device="cpu"):
    """
    Initializes a ResNet50 model structure, adapts the final layer for binary 
    classification, and loads the saved model weights from the specified path.

    Arguments:
        model_path: The file path to the saved model state dictionary.
        device: The computing device (CPU or GPU) to map the model weights to.
    """
    # Log the start of the initialization process
    print("Starting loading model...")
    
    # Initialize the ResNet50 architecture without pre-trained weights
    model = resnet50(weights=None)
    
    # Log the modification of the architecture
    print("Changing the final layer to a binary classification layer...")
    
    # Modify the fully connected layer to output two classes
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # Log the weight loading process
    print("Loading the model weights...")
    
    # Load the state dictionary from the disk to the specified device
    loaded_sd = torch.load(model_path, map_location=device)
    
    # Log the application of weights to the model
    print("Loading the model weights into the model...")
    
    # Apply the loaded state dictionary to the model architecture
    model.load_state_dict(loaded_sd, strict=False)
    
    # Log the successful completion
    print("\nModel loaded successfully!\n")
    
    return model


def display_model_architecture(model):
    """
    Traverses the provided PyTorch model and displays its architecture as a 
    styled HTML table, including hierarchy, layer types, and parameter counts.

    Arguments:
        model: The PyTorch model instance to inspect.
    """
    # Initialize a list to hold the dictionary data for each layer
    layer_data = []
    
    # Iterate through the immediate child modules of the network
    for name, module in model.named_children():
        # Calculate the total number of parameters in the current module
        params = sum(p.numel() for p in module.parameters())
        
        # Determine the layer type string
        if isinstance(module, torch.nn.Conv2d):
            layer_type = f"Conv2d {module.kernel_size}"
        else:
            layer_type = module.__class__.__name__
            
        # Append the parent layer information to the data list
        layer_data.append({
            "Layer Hierarchy": f"<b>{name}</b>",
            "Layer Type": layer_type,
            "Parameters": f"{params:,}"
        })
        
        # Check if the module is a container (Sequential) to inspect sub-layers
        if isinstance(module, torch.nn.Sequential):
            # Iterate through the sub-layers within the sequential block
            for sub_name, sub_module in module.named_children():
                # Calculate parameters for the sub-module
                sub_params = sum(p.numel() for p in sub_module.parameters())
                
                # Determine the sub-layer type string
                if isinstance(sub_module, torch.nn.Conv2d):
                    sub_type = f"Conv2d {sub_module.kernel_size}"
                else:
                    sub_type = sub_module.__class__.__name__
                
                # Append the sub-layer information with visual indentation
                layer_data.append({
                    "Layer Hierarchy": f"&nbsp;&nbsp;&nbsp;&nbsp;└─ {sub_name}",
                    "Layer Type": sub_type,
                    "Parameters": f"{sub_params:,}"
                })

    # Create a Pandas DataFrame from the collected layer data
    df = pd.DataFrame(layer_data)

    # Apply styling to the DataFrame for a clean HTML presentation
    styler = df.style.hide(axis="index")
    styler.set_table_styles([
        {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse"), ("font-family", "sans-serif")]},
        {"selector": "th", "props": [
            ("text-align", "left"), ("padding", "10px"), 
            ("background-color", "#4f4f4f"), ("color", "white"),
            ("border-bottom", "1px solid #ddd")
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"), ("padding", "8px"), 
            ("border-bottom", "1px solid #ddd")
        ]},
    ])
    
    # Render the styled table to HTML
    table_html = styler.to_html()

    # Calculate the total number of parameters in the entire network
    total_params = sum(p.numel() for p in model.parameters())
    
    # Create a summary HTML block for total parameters
    summary_html = f"""
    <div style="margin-top: 15px; font-family: monospace; font-size: 1.1em;">
        <p><b>Total Parameters:</b> {total_params:,}</p>
        <hr>
    </div>
    """

    # Combine the table and summary HTML
    final_html = table_html + summary_html
    
    # Display the final HTML output in the notebook
    display(HTML(final_html))


def predict_fruit_quality(model, root_dir, device="cpu"):
    """
    Creates an interactive widget to select a fruit subdirectory and 
    displays a 2x5 grid of predictions for images in that directory 
    using an Output widget for stable rendering.

    Arguments:
        model: The trained PyTorch model to be used for inference.
        root_dir: The root directory path containing the fruit image subdirectories.
        device: The computing device to perform inference on (default is "cpu").
    """
    
    # Scan the root directory for valid subdirectories
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return

    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    options = ['Select directory'] + subdirs

    # Initialize the dropdown widget for directory selection
    dropdown = widgets.Dropdown(
        options=options,
        value='Select directory',
        description='Directory:',
        disabled=False,
    )
    
    # Initialize the output widget to handle plot rendering
    output_widget = widgets.Output()

    # Define the callback function to update the grid based on dropdown selection
    def update_grid(change):
        folder_name = change['new'] if isinstance(change, dict) else change
        
        # Use the output widget context manager to ensure plots render inside the widget
        with output_widget:
            # Clear the previous output before rendering new plots
            output_widget.clear_output(wait=True) 
            
            if folder_name == 'Select directory':
                return

            folder_path = os.path.join(root_dir, folder_name)
            image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
            image_files = image_files[:10]
            
            if "Healthy" in folder_name:
                true_label_idx = 0
                true_label_str = "Fresh/Healthy"
            else:
                true_label_idx = 1
                true_label_str = "Rotten"

            # Initialize the matplotlib figure and axes
            # Set figure dimensions to accommodate text and spacing
            fig, axes = plt.subplots(2, 5, figsize=(18, 10))
            fig.suptitle(f"Predictions for: {folder_name}", fontsize=18, y=0.98)
            axes = axes.flatten()

            model.eval()

            # Iterate through the axes and display images with predictions using a progress bar
            for i, ax in tqdm(enumerate(axes), total=len(axes), desc="Analyzing images", unit="img"):
                if i < len(image_files):
                    img_path = os.path.join(folder_path, image_files[i])
                    input_tensor = preprocess_image(img_path, device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, predicted_idx = torch.max(outputs, 1)
                    
                    predicted_idx = predicted_idx.item()
                    pred_label_str = "Fresh/Healthy" if predicted_idx == 0 else "Rotten"

                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    ax.axis('off')

                    is_correct = (predicted_idx == true_label_idx)
                    color = 'green' if is_correct else 'red'
                    
                    # Configure the text display for labels
                    # Set the true label as the title with padding
                    ax.set_title(f"True: {true_label_str}", color='black', fontsize=13, pad=25)
                    
                    # Display the predicted label above the image
                    # Position the text slightly above the axis to avoid overlap
                    ax.text(0.5, 1.05, f"Pred: {pred_label_str}", 
                            color=color, fontsize=13, weight='bold', 
                            ha='center', transform=ax.transAxes)
                else:
                    ax.axis('off')

            # Adjust layout spacing to prevent overlap between rows and columns
            plt.subplots_adjust(hspace=0.5)
            plt.tight_layout()
            plt.show()

    # Attach the observer to the dropdown widget to trigger updates on value change
    dropdown.observe(update_grid, names='value')

    # Display the interactive widgets
    display(dropdown, output_widget)


def preprocess_image(image_path, device="cpu"):
    """
    Loads and transforms an image file into a normalized tensor suitable for model inference.

    Arguments:
        image_path: The file path to the input image.
        device: The computing device (CPU or GPU) to place the tensor on.
    """
    # Define the sequence of image transformations including resizing, cropping, and normalization
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    # Load the image, apply the transformations, add a batch dimension, and transfer to the specified device
    image_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    return image_tensor


def grid_from_layer(feat, num_filters=4):
    """
    Creates a tiled grid image from a subset of channels in a feature map.

    This function selects a specified number of filters from the input feature
    tensor at equal intervals and arranges them into a square grid for visualization.
    The resulting grid is normalized to the [0, 1] range.

    Arguments:
        feat: The input feature map tensor (typically from a CNN layer).
        num_filters: The number of channels to select and display in the grid.
    """
    # Remove single-dimensional entries, move to CPU, and convert to a NumPy array
    feat = feat.squeeze().cpu().numpy()
    
    # Unpack the channel, height, and width dimensions
    C, H, W = feat.shape
    
    # Select indices for the filters to visualize, spaced equally across the channels
    idxs = np.linspace(0, C - 1, num_filters, dtype=int)

    # Calculate the side length of the square grid needed to hold the filters
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    # Initialize an empty array for the final grid image
    grid = np.zeros((grid_size * H, grid_size * W))

    # Iterate through the selected filter indices to populate the grid
    for i, c in enumerate(idxs):
        # Calculate row and column positions for the current filter
        r, col = divmod(i, grid_size)
        
        # Insert the feature map into the calculated position within the grid
        grid[
            r * H : (r + 1) * H, col * W : (col + 1) * W
        ] = feat[c]

    # Shift values so the minimum is 0
    grid -= grid.min()
    
    # Scale values so the maximum is 1, adding a small epsilon to prevent division by zero
    grid /= grid.max() + 1e-5
    
    return grid


def display_feature_hierarchy(activations, img):
    """
    Visualizes the original image alongside a grid of feature maps from 
    selected layers of a convolutional neural network.

    This function de-normalizes the input image for display and creates a 
    multi-panel plot showing the original image and the aggregated feature 
    activations for specific layers (conv1 through layer4).

    Arguments:
        activations: A dictionary containing feature map tensors for specific layers.
        img: The preprocessed input image tensor (normalized).
    """
    # Create the main figure with specific dimensions
    plt.figure(figsize=(15, 10))

    # Move the image tensor to CPU and prepare for de-normalization
    orig = img[0].cpu()
    
    # Reverse the standard ImageNet normalization (multiply by std)
    orig = orig * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Reverse the standard ImageNet normalization (add mean)
    orig = orig + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    # Create the first subplot for the original image
    plt.subplot(2, 3, 1)
    
    # Permute tensor dimensions from (C, H, W) to (H, W, C) and clamp values to [0, 1] for display
    plt.imshow(orig.permute(1, 2, 0).clamp(0, 1))
    
    # Set the title for the original image plot
    plt.title("Original Image")
    
    # Hide the axis ticks and labels
    plt.axis("off")

    # Define the specific order of layers to visualize
    order = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    
    # Iterate through the layers to create subplots, starting from the second position
    for sp, name in enumerate(order, start=2):
        # Generate a grid visualization from the layer's activation tensor
        grid = grid_from_layer(activations[name])
        
        # Select the specific subplot position
        plt.subplot(2, 3, sp)
        
        # Display the feature grid using the viridis colormap
        plt.imshow(grid, cmap="viridis")
        
        # Set the title including the layer name and number of filters
        plt.title(f"{name}: {activations[name].shape[1]} filters")
        
        # Hide the axis ticks and labels
        plt.axis("off")

    # Adjust the layout to prevent overlapping elements
    plt.tight_layout()
    
    # Render the final figure
    plt.show()


def visual_strip(upsampled):
    """
    Displays a horizontal strip of upsampled feature maps.

    This function concatenates a list of image tensors into a single grid 
    row and renders it using Matplotlib to visualize the progression 
    of features across model layers.

    Arguments:
        upsampled: A list of 4D tensors (Batch, Channel, Height, Width) 
                   representing processed feature maps.
    """
    # Create a single image grid from the list of tensors, arranged in one row with padding
    grid = make_grid(torch.cat(upsampled), nrow=5, padding=2)
    
    # Initialize the figure with a wide aspect ratio suitable for a horizontal strip
    plt.figure(figsize=(14, 3))
    
    # Transpose the tensor dimensions from (C, H, W) to (H, W, C) and move to CPU for Matplotlib compatibility
    plt.imshow(grid.cpu().permute(1, 2, 0))
    
    # Hide the axis ticks and labels
    plt.axis("off")
    
    # Set the descriptive title for the plot
    plt.title("Feature-map progression conv1 → layer4")
    
    # Render the final figure
    plt.show()


def display_saliency(image_tensor, heatmap, alpha=0.5):
    """
    Overlays a saliency heatmap onto the original image for visualization.

    This function denormalizes the input image (assuming ImageNet statistics),
    clamps pixel values, and displays the heatmap using a jet colormap
    with transparency.

    Arguments:
        image_tensor: The input image tensor (1, 3, H, W).
        heatmap: The 2D saliency map tensor (H, W).
        alpha: The transparency factor for the heatmap overlay (0.0 to 1.0).
    """
    # Define the mean used for ImageNet normalization for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # Define the standard deviation used for ImageNet normalization
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize the image tensor by multiplying by std and adding mean
    img_disp = image_tensor.detach()[0].cpu() * std + mean
    
    # Clamp values to [0, 1] range, rearrange dimensions to (H, W, C), and convert to numpy
    img_disp = img_disp.clamp(0, 1).permute(1, 2, 0).numpy()

    # Initialize the plot figure with specific dimensions
    plt.figure(figsize=(6, 6))
    
    # Display the base image
    plt.imshow(img_disp)
    
    # Overlay the heatmap with the specified colormap and transparency
    plt.imshow(heatmap.cpu(), cmap="jet", alpha=alpha)
    
    # Hide the axis ticks and labels
    plt.axis("off")
    
    # Set the title for the visualization
    plt.title("Salience Map Overlay")
    
    # Render the final plot
    plt.show()


def display_cam(image_tensor, cam_up):
    """
    Overlays a Class Activation Map (CAM) onto the original image.

    This function prepares the input image by reversing the normalization 
    process (assuming ImageNet statistics), and then renders the image 
    with the CAM heatmap superimposed to visualize regions of interest.

    Arguments:
        image_tensor: The input image tensor (1, 3, H, W).
        cam_up: The upsampled CAM heatmap tensor (H, W).
    """
    # Define the mean values used for ImageNet normalization
    means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # Define the standard deviation values used for ImageNet normalization
    stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize the image: multiply by std, add mean, clamp to [0, 1], and rearrange to (H, W, C)
    rgb = (image_tensor[0].cpu() * stds + means).clamp(0, 1).permute(1, 2, 0).numpy()

    # Initialize the figure with specific dimensions
    plt.figure(figsize=(6, 6))
    
    # Display the base RGB image
    plt.imshow(rgb)
    
    # Overlay the CAM heatmap using the jet colormap and 50% transparency
    plt.imshow(cam_up.cpu().detach().cpu(), cmap="jet", alpha=0.5)
    
    # Hide the axis ticks and labels
    plt.axis("off")
    
    # Set the title for the visualization
    plt.title("Simplified CAM Overlay")
    
    # Render the final figure
    plt.show()


def check_model_snapshot():
    """
    Verifies the integrity of the model snapshot directory.

    Checks if the snapshot directory exists and is not empty. If the directory
    is invalid, it attempts to locate and extract a zip archive containing the
    snapshots, filtering out unnecessary system files during the process.
    """
    # Define the base paths for the model and the snapshots
    base_path = "./models/models--stabilityai--stable-diffusion-2-base/"
    snapshot_dir = os.path.join(base_path, "snapshots")
    zip_path = os.path.join(base_path, "snapshots.zip")

    # Check if the snapshots directory exists and is populated
    if os.path.exists(snapshot_dir) and len(os.listdir(snapshot_dir)) > 0:
        print("Snapshots directory exists and is not empty. Skipping extraction.")
        return

    # Ensure the zip file exists if the directory is missing or empty
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Snapshots directory is missing/empty and zip file not found at: {zip_path}")

    print("Prepare to unzip snapshots...")

    # Open the zip file in read mode
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Retrieve the list of all file names in the archive
        all_members = zip_ref.namelist()

        # Filter out macOS specific system artifacts to ensure a clean extraction
        # This is done prior to iteration to ensure the progress bar total is accurate
        valid_members = [
            m for m in all_members 
            if not m.startswith("__MACOSX") and not m.endswith(".DS_Store")
        ]

        # Iterate through the valid files and extract them to the base path
        for member in tqdm(valid_members, desc="Unzipping", unit="file"):
            zip_ref.extract(member, base_path)

    print("Extraction complete.")