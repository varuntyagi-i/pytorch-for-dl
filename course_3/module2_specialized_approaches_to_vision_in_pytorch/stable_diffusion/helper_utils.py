import io
import os
import time
import zipfile

import imageio
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMPipeline, StableDiffusionPipeline
from IPython.display import display
from PIL import Image
from tqdm.auto import tqdm


def check_model_snapshot(model_cache_path):
    """
    Verifies the integrity of the model snapshot directory and performs extraction if needed.

    Checks if the snapshot directory exists and is populated. If the directory is 
    invalid or empty, it locates the corresponding zip archive and extracts the 
    contents, filtering out system artifacts during the process.

    Arguments:
        model_cache_path: The directory path where the model artifacts should be located.
    """
    # Define the base root path for the model directory
    base_path = model_cache_path + "/"
    
    # Construct the path for the snapshots subdirectory
    snapshot_dir = os.path.join(base_path, "snapshots")
    
    # Construct the path for the backup zip archive
    zip_path = os.path.join(base_path, "snapshots.zip")

    # Verify if the snapshots directory already exists and contains files
    if os.path.exists(snapshot_dir) and len(os.listdir(snapshot_dir)) > 0:
        print("Snapshots directory exists and is not empty. Skipping extraction.")
        return

    # Raise an error if the required zip file is not found when snapshots are missing
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Snapshots directory is missing/empty and zip file not found at: {zip_path}")

    print("Prepare to unzip snapshots...")

    # Open the zip archive for reading
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Retrieve the list of all files contained in the zip archive
        all_members = zip_ref.namelist()

        # Filter out system artifacts and macOS specific files from the extraction list
        valid_members = [
            m for m in all_members 
            if not m.startswith("__MACOSX") and not m.endswith(".DS_Store")
        ]

        # Iterate through the validated file list and extract them to the base path
        for member in tqdm(valid_members, desc="Unzipping", unit="file"):
            zip_ref.extract(member, base_path)

    print("Extraction complete.")


def display_images(images, titles=None, figsize=None):
    """
    Displays one or more images in a row with optional titles.
    
    Args:
        images: A single image or a list of images.
        titles: A list of titles corresponding to the images (optional).
        figsize: A tuple (width, height) for the figure. 
                 If None, defaults to (5 * num_images, 5).
    """
    # 1. Normalize input: Ensure 'images' is always a list
    if not isinstance(images, (list, tuple)):
        images = [images]
    
    num_images = len(images)
    
    # 2. Smart Figure Size: Default to 5 inches wide per image if not specified
    if figsize is None:
        figsize = (5 * num_images, 5)

    # 3. Create Subplots
    fig, axs = plt.subplots(1, num_images, figsize=figsize)
    
    # 4. Handle Single Axis: Make 'axs' iterable even if there is only 1 image
    if num_images == 1:
        axs = [axs]
        
    # 5. Plot Loop
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
        
        # Add title if provided and index exists
        if titles and i < len(titles):
            axs[i].set_title(titles[i])

    print() # To add space
    plt.tight_layout()
    plt.show()


def display_grid(images, titles=None, n_rows=None, n_cols=None, figsize=None, main_title=None, row_labels=None):
    """
    Displays a grid of images. Handles both flat lists (wrapping) and lists of lists (defined rows).

    Args:
        images: Either a flat list of images (requires n_rows/n_cols) 
                OR a list of lists of images (e.g. [row1_images, row2_images]).
        titles: Optional list of strings for subplot titles. 
                Should match the total number of images (flattened order).
        n_rows: Number of rows (required if images is a flat list).
        n_cols: Number of columns (required if images is a flat list).
        figsize: Tuple (width, height). Defaults to auto-calculated based on grid size.
        main_title: Super title for the entire figure.
        row_labels: Optional list of strings for row labels (left side).
    """
    # 1. Detect Input Type: Is it a flat list or a list of rows?
    if isinstance(images[0], list): 
        # Input is a list of lists (Snippet 2 style)
        n_rows = len(images)
        n_cols = len(images[0])
        flat_images = [img for row in images for img in row]
    else: 
        # Input is a flat list (Snippet 1 style)
        flat_images = images
        if n_rows is None or n_cols is None:
            raise ValueError("For flat lists, n_rows and n_cols must be provided.")

    # 2. Setup Figure
    if figsize is None:
        # Heuristic: 3 inches per column, 3.5 inches per row
        figsize = (3 * n_cols, 3.5 * n_rows)
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if main_title:
        fig.suptitle(main_title, fontsize=16)

    # 3. Normalize Axes: Ensure axes is always a 2D array [row, col]
    # This handles edge cases like 1xN or Nx1 grids automatically
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    # 4. Plotting Loop
    for i, img in enumerate(flat_images):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        
        ax.imshow(img)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)

    # 5. Add Row Labels (if provided)
    if row_labels:
        for row_idx, label in enumerate(row_labels):
            if row_idx < n_rows:
                # Add label to the first column of the row
                axes[row_idx, 0].set_ylabel(label, rotation=90, size='large', labelpad=20)
                # Note: For labels to show when axis is 'off', we might need specific matplotlib backend 
                # behavior or manual text. If using standard styles, keeping the axis off might hide this.
                # However, this preserves the logic from your original Snippet 2.

    print() # To add space
    plt.tight_layout()
    plt.show()


def remove_all_noise_at_once(noisy_image, predicted_noise, timestep, scheduler):
    """
    Apply the diffusion denoising formula to estimate the original clean image (x_0)
    directly from the current noisy state in a single step.

    Arguments:
        noisy_image: The current latent tensor with noise.
        predicted_noise: The noise component predicted by the model.
        timestep: The current diffusion timestep.
        scheduler: The noise scheduler containing alpha schedules.

    Returns:
        The estimated clean image tensor.
    """
    # Retrieve the cumulative alpha value for the specific timestep
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(noisy_image.device)
    
    # Compute the square root of the cumulative alpha
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)

    # Compute the square root of one minus the cumulative alpha
    sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
    
    # Apply the mathematical formula to subtract the predicted noise and scale the result
    clean_image = (noisy_image - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
    
    # Return the resulting estimated clean image
    return clean_image


def gradual_denoise_step(noisy_image, predicted_noise, timestep, scheduler):
    """
    Perform a single iteration of the denoising process to transition from
    timestep t to t-1.

    Arguments:
        noisy_image: The current image tensor.
        predicted_noise: The noise predicted by the model.
        timestep: The current step in the diffusion process.
        scheduler: The scheduler object handling the step logic.

    Returns:
        The slightly denoised image sample for the next step.
    """
    # Execute the scheduler step function to compute the previous sample
    step = scheduler.step(predicted_noise, timestep, noisy_image)
    
    # Extract and return the denoised sample from the scheduler output
    return step.prev_sample


@torch.no_grad()
def tensor_to_image(tensor):
    """
    Process a raw PyTorch tensor into a viewable PIL Image object.
    
    Arguments:
        tensor: The input tensor representing an image.

    Returns:
        A PIL Image object scaled to 0-255.
    """
    # Move tensor to CPU, remove batch dimension, and reorder dimensions to HxWxC
    img = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Normalize values to the range [0, 1] if the input range is [-1, 1]
    img = (img + 1) / 2 if img.min() < 0 else img
    
    # Ensure all pixel values fall strictly within the valid range
    img = np.clip(img, 0, 1)
    
    # Scale values to 0-255, cast to unsigned 8-bit integers, and convert to PIL format
    return Image.fromarray((img * 255).astype(np.uint8))


@torch.no_grad()
def visualize_ddpm_denoising(pipe, num_inference_steps=100):
    """
    Execute the diffusion process to visualize both gradual denoising and 
    one-step noise removal approximations at various intervals.

    Arguments:
        pipe: The stable diffusion pipeline object.
        num_inference_steps: The total number of denoising steps to perform.

    Returns:
        A tuple containing two lists: gradual denoising images and full removal approximation images.
    """
    # Extract image configuration parameters from the model configuration
    image_size = pipe.unet.config.sample_size
    num_channels = pipe.unet.config.in_channels
    device = pipe.device

    # Initialize a random noise tensor to serve as the starting point
    images = torch.randn(1, num_channels, image_size, image_size, device=device)
    
    # distinct references to the scheduler and UNet model
    scheduler = pipe.scheduler
    model = pipe.unet
    
    # Configure the scheduler with the specified number of inference steps
    scheduler.set_timesteps(num_inference_steps)

    # Initialize a list to track specific timesteps for visualization
    timesteps_to_show = []
    n_vis = 7

    # Calculate indices for the steps to be visualized
    for i in range(n_vis):
        # Determine the index for the current visualization interval
        t_idx = int((num_inference_steps - 1) * i / (n_vis - 1))
        timesteps_to_show.append(t_idx)

    # Initialize storage lists for the output images
    gradual_images = []     # stores (step_index, gradual denoised image)
    full_removal_images = [] # stores (step_index, "full denoise" image)
    
    # Set up the timesteps and initial latent variables
    timesteps = scheduler.timesteps
    latents = images

    # Iterate through each timestep in the denoising process
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Generate a noise prediction using the model
        noise_pred = model(latents, t).sample

        # Determine if the current step is flagged for visualization
        if i in timesteps_to_show:
            # Estimate the fully clean image from the current state
            full_removal = remove_all_noise_at_once(latents, noise_pred, t, scheduler)
            full_removal_img = tensor_to_image(full_removal)
            full_removal_images.append((i, full_removal_img))
            
            # Convert the current noisy latent to an image for comparison
            current_img = tensor_to_image(latents)
            gradual_images.append((i, current_img))

        # Update the latents by performing a single denoising step
        latents = gradual_denoise_step(latents, noise_pred, t, scheduler)

    # Convert the final resulting tensor to an image
    final_img = tensor_to_image(latents)
    gradual_images.append((num_inference_steps, final_img))

    # Return the collected image sequences
    return gradual_images, full_removal_images


def load_widget(pipe):
    """
    Initialize and display an interactive UI widget for image generation.

    Arguments:
        pipe: The loaded Stable Diffusion pipeline.
    """
    
    # Initialize output container and image display widgets
    output = widgets.Output()
    gif_widget = widgets.Image(format='gif', width=320, height=320)
    final_image_widget = widgets.Image(format='png', width=320, height=320)
    
    # Define generic styling dictionaries for widgets
    textbox_style = {'description_width': '120px'}
    slider_style = {'description_width': '140px'}
    
    # Create the HTML header element
    heading = widgets.HTML("<h3 style='color:#0EA5E9;font-family:sans-serif'>Stable Diffusion Image Generator</h3>")
    
    # Initialize the checkbox for toggling animation mode
    mode_checkbox = widgets.Checkbox(
        value=True,
        description='Show denoising animation (GIF)',
        indent=False,
        layout=widgets.Layout(width='300px')
    )
    
    # Create an informational tooltip/label regarding the generation mode
    mode_info = widgets.HTML(
        value="""<div style='background-color:#000000;padding:8px;border-radius:6px;margin:8px 0;'>
        <small><b>💡 Tip:</b> Uncheck for faster generation (final image only)</small>
        </div>"""
    )
    
    # Create the text input field for the primary prompt
    prompt_widget = widgets.Text(
        value="A puppy dog riding a skateboard in times square", 
        description='Prompt:', 
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )
    
    # Create the text input field for the negative prompt
    negative_prompt_widget = widgets.Text(
        value="", 
        description='Negative:', 
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )
    
    # Create a slider for adjusting the number of inference steps
    steps_slider = widgets.IntSlider(
        value=50, min=10, max=100, step=1,
        description='Inference steps:', 
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    
    # Create a label to display the current steps value
    steps_value = widgets.Label(f"{steps_slider.value}")
    
    # Define a callback to update the steps label when the slider moves
    def update_steps_label(*a):
        steps_value.value = f"{steps_slider.value}"
    steps_slider.observe(update_steps_label, "value")
    
    # Create a slider for adjusting the guidance scale
    gs_slider = widgets.FloatSlider(
        value=7.5, min=4.0, max=16.0, step=0.1,
        description='Guidance scale:', 
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    
    # Create a label to display the current guidance scale value
    gs_value = widgets.Label(f"{gs_slider.value:.2f}")
    
    # Define a callback to update the guidance scale label when the slider moves
    def update_gs_label(*a):
        gs_value.value = f"{gs_slider.value:.2f}"
    gs_slider.observe(update_gs_label, "value")
    
    # Initialize the execution button
    run_button = widgets.Button(
        description="✨ Generate with Animation ✨", 
        button_style='info',
        layout=widgets.Layout(width='250px', height='40px')
    )
    
    # Define logic to update button text based on the selected mode
    def update_button_text(*args):
        if mode_checkbox.value:
            run_button.description = "✨ Generate with Animation ✨"
            run_button.button_style = 'info'
        else:
            run_button.description = "⚡ Quick Generate ⚡"
            run_button.button_style = 'success'
    mode_checkbox.observe(update_button_text, 'value')
    
    # Arrange the input widgets vertically
    input_form = widgets.VBox([
        heading,
        widgets.HBox([mode_checkbox, mode_info]),
        prompt_widget,
        negative_prompt_widget,
        widgets.HBox([steps_slider, steps_value]),
        widgets.HBox([gs_slider, gs_value]),
        run_button,
    ], layout=widgets.Layout(
        border='2px solid #0EA5E9', 
        box_shadow="2px 2px 10px #0EA5E9",
        padding='18px 24px 18px 24px', 
        border_radius='14px',
        width='600px',
        background='white'
    ))
    
    # Group the image widgets into a container
    image_container = widgets.VBox([gif_widget, final_image_widget])
    
    # Combine the form, output log, and images into the main UI object
    ui = widgets.VBox([
        input_form,
        output,
        image_container
    ])
    
    def generate_with_animation(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """
        Execute generation process including intermediate step capture for animation.

        Arguments:
            prompt: Text prompt for generation.
            negative_prompt: Negative text prompt.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale value.
        """
        with output:
            # Clear previous output logs
            output.clear_output()
            print(f"\n🎬 Generating animation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()
            
            # Initialize storage for captured frames
            intermediate_images = []
            
            # Initialize and display the progress bar
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=num_inference_steps,
                description='Denoising:',
                bar_style='info',
                orientation='horizontal'
            )
            display(progress)
            
            # Define the callback function to run at each step
            def collect_callback(step, timestep, latents):
                with torch.no_grad():
                    # Scale latents to match the VAE expectation
                    scaled_latents = latents / 0.18215
                    # Decode latents to image space
                    image = pipe.vae.decode(scaled_latents).sample
                    # Normalize and clamp image values
                    image = (image / 2 + 0.5).clamp(0, 1)
                    # Rearrange dimensions for PIL compatibility
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    # Convert to PIL Image object
                    pil_img = Image.fromarray((image * 255).astype('uint8'))
                # Store the processed image frame
                intermediate_images.append((step, pil_img))
                # Update progress bar value
                progress.value = step + 1
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Create a deterministic generator based on a fixed seed
            generator = torch.Generator(pipe.device).manual_seed(42)
            intermediate_images.clear()
            
            # Execute the pipeline with the callback function
            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=collect_callback,
                callback_steps=1,
                progress_bar=False,
            )
            
            # Update visual indicators for completion
            progress.bar_style = 'success'
            progress.description = 'Done'
            
            print("\n🎨 Creating animation...")
            # Initialize buffer for GIF data
            gif_bytes = io.BytesIO()
            # Extract image objects from the storage list
            all_imgs = [img for (step, img) in intermediate_images]
            # Save the sequence of images as a GIF
            imageio.mimsave(gif_bytes, all_imgs, format='GIF', duration=0.15)
            
            # Update widget visibility to show the animation
            final_image_widget.layout.display = 'none'
            gif_widget.layout.display = 'block'
            gif_widget.value = gif_bytes.getvalue()
            
            # Calculate and print elapsed time
            elapsed = time.time() - start_time
            print(f"✅ Animation generated in {elapsed:.1f} seconds")
    
    def generate_final_only(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """
        Execute generation process returning only the final result for efficiency.

        Arguments:
            prompt: Text prompt for generation.
            negative_prompt: Negative text prompt.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale value.
        """
        with output:
            # Clear previous output logs
            output.clear_output()
            print(f"\n⚡ Quick generation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()
            
            # Initialize and display the progress bar
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=1,
                description='Generating:',
                bar_style='success',
                orientation='horizontal'
            )
            display(progress)
            
            # Record start time
            start_time = time.time()
            # Create a deterministic generator
            generator = torch.Generator(pipe.device).manual_seed(42)
            
            # Run the pipeline without callbacks
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                progress_bar=False,
            )
            
            # Update progress bar to completion
            progress.value = 1
            progress.description = 'Done'
            
            # Retrieve the resulting image
            final_img = result.images[0]
            # Save image to a byte buffer
            img_bytes = io.BytesIO()
            final_img.save(img_bytes, format='PNG')
            
            # Update widget visibility to show the static image
            gif_widget.layout.display = 'none'
            final_image_widget.layout.display = 'block'
            final_image_widget.value = img_bytes.getvalue()
            
            # Calculate and print elapsed time
            elapsed = time.time() - start_time
            print(f"✅ Image generated in {elapsed:.1f} seconds")
            
            # Display summary statistics
            print(f"\n📊 Generation Stats:")
            print(f"  • Mode: Quick (final image only)")
            print(f"  • Steps: {num_inference_steps}")
            print(f"  • Guidance Scale: {guidance_scale}")
            print(f"  • Time saved: ~{num_inference_steps * 0.15:.1f}s (animation creation)")
    
    # Define the click handler for the run button
    def run_on_click(b):
        if mode_checkbox.value:
            generate_with_animation(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )
        else:
            generate_final_only(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )
    
    # Attach the click handler to the button
    run_button.on_click(run_on_click)
    # Render the UI
    display(ui)