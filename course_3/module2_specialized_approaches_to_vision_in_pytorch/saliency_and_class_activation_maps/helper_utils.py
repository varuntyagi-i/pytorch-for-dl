import os
import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_widget(compute_gradcam, visualize_gradcam, model, transform, device, folder="images"):
    """
    Create and display an interactive widget to visualize GradCAM results for images.

    This function scans a directory for images and generates a dropdown menu. 
    Selecting an image triggers the computation and display of the GradCAM heatmap 
    using the provided model and visualization functions.

    Args:
        compute_gradcam: A callable that takes an image path, model, transform, and device, returning results.
        visualize_gradcam: A callable that displays the image, heatmap, and prediction details.
        model: The pre-trained PyTorch model to use for inference.
        transform: The preprocessing transform pipeline to apply to input images.
        device: The compute device (CPU or GPU) where the model is loaded.
        folder: The directory path containing the input images.
    """
    def get_jpg_files(folder=folder):
        """
        Retrieve a sorted list of JPEG filenames from the specified directory.

        Args:
            folder: The directory path to search for images.

        Returns:
            A list of strings representing filenames ending in .jpg.
        """
        # Check if the specified folder exists
        if not os.path.exists(folder):
            return []
        # Return a sorted list of files that match the jpg extension, case-insensitive
        return sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

    # Initialize a widget output area to capture and display plots
    out = widgets.Output()

    def gradcam_widget_view(image_name):
        """
        Handle the update logic when a new image is selected from the dropdown.

        This callback runs the GradCAM computation pipeline and updates the 
        output widget with the new visualization.

        Args:
            image_name: The name of the selected image file.
        """
        # Construct the full file path for the selected image
        img_path = os.path.join(folder, image_name)
        # Use the output widget context to capture prints and plots
        with out:
            # Clear the previous output to prepare for the new plot
            clear_output(wait=True)
            # Display a status message indicating processing has started
            print(f"Showing GradCAM for: {image_name}")
            # Run the provided GradCAM computation function
            img_display, heatmap, pred_class, pred_score = compute_gradcam(
                img_path, model, transform, device
            )
            # Verify if the image processing was successful
            if img_display is not None:
                # Extract the filename without extension to use as a title
                title = os.path.splitext(image_name)[0]
                # Render the visualization using the provided helper function
                visualize_gradcam(img_display, heatmap, pred_class, pred_score, title)
            else:
                # Log an error message if processing failed
                print(f"Could not process {image_name}.")

    # Fetch the list of available images from the folder
    jpg_list = get_jpg_files(folder)
    # Check if the list is empty and display a warning if so
    if not jpg_list:
        display(widgets.HTML(
            value=f"<b style='color: red;'>No .jpg files found in the <code>{folder}/</code> folder. Please add images and rerun.</b>"
        ))
    else:
        # Create a dropdown menu widget with the list of image files
        dropdown = widgets.Dropdown(
            options=jpg_list,
            description='Select image:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%'),  # Set width for better visibility
            disabled=False
        )
        # Display the header instructions
        display(widgets.HTML(f"<h3>GradCAM Visualizer</h3>Select an image from <code>{folder}</code>:"))
        # Link the dropdown widget to the callback function to trigger updates
        widgets.interact(gradcam_widget_view, image_name=dropdown)
        # Display the output container where the plots will appear
        display(out)