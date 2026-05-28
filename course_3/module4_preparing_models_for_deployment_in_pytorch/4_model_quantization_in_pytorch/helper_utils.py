import os
import tempfile
import time

from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from transformers import BlipForQuestionAnswering, BlipProcessor


def load_cifar10():
    """
    Loads the CIFAR10 dataset, applies transformations, and initializes data loaders.

    Args:

    Returns:
        trainloader (torch.utils.data.DataLoader): The data loader for the training dataset 
                                                   containing augmented images.
        testloader (torch.utils.data.DataLoader): The data loader for the testing dataset 
                                                  containing normalized images.
    """
    # Define a series of transformations to apply to the training images.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Define a simpler set of transformations for the test images.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Specify the local directory to store the dataset.
    data_path = './CIFAR10_data'
    
    # Check if the data directory already exists to avoid re-downloading.
    if os.path.exists(data_path) and os.path.isdir(data_path):
        # If the folder exists, set download to False.
        download = False  
        print("CIFAR10 Data folder found locally. Loading from local.\n")
    else:
        # If the folder doesn't exist, set download to True.
        download = True   
        print("CIFAR10 Data folder not found locally. Downloading data.\n")

    # Set the number of images to be processed in one batch.
    batch_size = 128

    # Load the CIFAR10 training dataset and apply the defined training transformations.
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=download, transform=transform_train)
    
    # Create a data loader for the training set, which will shuffle the data and serve it in batches.
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load the CIFAR10 test dataset and apply the defined test transformations.
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=download, transform=transform_test)
    
    # Create a data loader for the test set. Shuffling is not necessary for the test set.
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Return the training and testing data loaders.
    return trainloader, testloader
    


def training_loop(model, trainloader, testloader, num_epochs, DEVICE):
    """
    Trains and validates a neural network model over a specified number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to train.
        trainloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        testloader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        num_epochs (int): The total number of iterations over the dataset.
        DEVICE (torch.device): The hardware device to perform computations on.

    Returns:
        None: The function saves model checkpoints to disk and prints progress to the console.
    """

    # Set the model to the configured device for training
    model = model.to(DEVICE)
    
    # Define the loss function for multi-class classification
    loss_function = nn.CrossEntropyLoss()
    
    # Initialize the Adam optimizer with a fixed learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set up a learning rate scheduler that scales down the LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Initialize the tracking variable for the highest recorded accuracy
    best_accuracy = 0.0

    # Initialize the primary progress bar for the epoch loop
    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_loop:
        # Set the model to training mode
        model.train()
        
        # Reset the cumulative training loss for the current epoch
        train_loss = 0.0

        # Initialize the progress bar for the training batches
        train_inner_loop = tqdm(trainloader, total=len(trainloader), leave=False, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")
        
        for data in train_inner_loop:
            # Unpack the image data and corresponding labels
            inputs, labels = data
            
            # Transfer the data and labels to the active computation device
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Clear previous gradients before the backward pass
            optimizer.zero_grad()
            
            # Perform a forward pass through the model
            outputs = model(inputs)
            
            # Calculate the loss based on model predictions
            loss = loss_function(outputs, labels)
            
            # Compute gradients via backpropagation
            loss.backward()
            
            # Update the model weights using the optimizer
            optimizer.step()

            # Accumulate the scalar loss value
            train_loss += loss.item()

            # Update the progress bar display with the current batch loss
            train_inner_loop.set_postfix(train_loss=loss.item())

        # Set the model to evaluation mode
        model.eval()
        
        # Initialize validation metrics
        val_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient computation for the validation phase
        with torch.no_grad():
            # Initialize the progress bar for validation batches
            test_inner_loop = tqdm(testloader, total=len(testloader), leave=False, desc=f"Epoch [{epoch + 1}/{num_epochs}] Validation")
            
            for data in test_inner_loop:
                # Unpack validation data
                images, labels = data
                
                # Transfer validation data to the active device
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Perform a forward pass for evaluation
                outputs = model(images)
                
                # Calculate the validation loss for the current batch
                loss = loss_function(outputs, labels)
                
                # Accumulate the total validation loss
                val_loss += loss.item()

                # Determine the predicted class with the highest probability
                _, predicted = torch.max(outputs.data, 1)
                
                # Track the total number of processed samples
                total += labels.size(0)
                
                # Track the number of correct predictions
                correct += (predicted == labels).sum().item()

                # Update the validation progress bar display
                test_inner_loop.set_postfix(val_loss=loss.item())

        # Calculate the average training loss across all batches
        avg_train_loss = train_loss / len(trainloader)
        
        # Calculate the average validation loss across all batches
        avg_val_loss = val_loss / len(testloader)
        
        # Calculate the overall classification accuracy as a percentage
        accuracy = 100 * correct / total

        # Log the statistical summary for the current epoch
        tqdm.write(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Check if the current model outperformed previous iterations
        if accuracy > best_accuracy:
            # Update the record for best accuracy
            best_accuracy = accuracy
            
            # Save the state dictionary of the top-performing model
            torch.save(model.state_dict(), 'cifar10_cnn_best.pt')
            
            # Notify the user of the new saved checkpoint
            tqdm.write(f'Best model saved with accuracy: {best_accuracy:.2f}%')

        # Adjust the learning rate based on validation performance
        scheduler.step(avg_val_loss)

        # Update the main progress bar with the latest metrics
        epoch_loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss, accuracy=f"{accuracy:.2f}%")

        # Insert a blank line for visual separation in the console output
        tqdm.write("")

    # Signal the completion of the training process
    print(f'Finished Training with best accuracy: {best_accuracy:.2f}%')

    # Save the final state of the model after all epochs
    torch.save(model.state_dict(), 'cifar10_cnn_final.pt')
    
    # Confirm the final save operation
    print('Final model saved!')



def train_qat(model_to_train, trainloader, device, epochs=5):
    """
    Performs Quantization-Aware Training (QAT) fine-tuning for a given model.

    Args:
        model_to_train (torch.nn.Module): The QAT-prepared model.
        trainloader (torch.utils.data.DataLoader): The data loader for training.
        device (torch.device): The device to train on.
        epochs (int): The number of epochs to train.

    Returns:
        model_to_train (torch.nn.Module): The fine-tuned model after QAT.
    """
    
    # Move the model to the specified hardware device
    model_to_train.to(device)
    
    # Set the model to training mode to enable dropout and batch normalization updates
    model_to_train.train()

    # Define the loss function for the training process
    loss_function = nn.CrossEntropyLoss()
    
    # Initialize the Stochastic Gradient Descent optimizer with momentum
    optimizer = optim.SGD(model_to_train.parameters(), lr=0.001, momentum=0.9)
    
    # Initialize the progress bar for the entire epoch sequence
    epoch_loop = tqdm(range(epochs), desc="QAT Training Progress")
    
    # Iterate through the specified number of training epochs
    for epoch in epoch_loop:
        # Initialize tracking variables for the current epoch
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize a nested progress bar for individual batches
        batch_loop = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        # Iterate through batches provided by the data loader
        for data in batch_loop:
            # Extract inputs and target labels from the batch
            inputs, labels = data
            
            # Transfer the input data and labels to the configured device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradients for the current optimization step
            optimizer.zero_grad()
            
            # Perform a forward pass through the model
            outputs = model_to_train(inputs)
            
            # Compute the loss based on model predictions and actual labels
            loss = loss_function(outputs, labels)
            
            # Calculate gradients through backpropagation
            loss.backward()
            
            # Update the model parameters based on the computed gradients
            optimizer.step()
            
            # Accumulate the running loss value
            running_loss += loss.item()

            # Determine the predicted class for accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            
            # Update the total count of processed samples
            total += labels.size(0)
            
            # Update the count of correctly predicted samples
            correct += (predicted == labels).sum().item()
            
            # Update the batch progress bar with the current loss value
            batch_loop.set_postfix(loss=f"{loss.item():.3f}")

        # Calculate the average loss over all training batches
        avg_epoch_loss = running_loss / len(trainloader)
        
        # Calculate the final training accuracy for the epoch
        train_accuracy = 100 * correct / total
        
        # Print the statistical summary for the current epoch
        print(f'Epoch {epoch + 1}, Training Loss: {avg_epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        
        # Update the main epoch progress bar with the average loss
        epoch_loop.set_postfix(avg_loss=f"{avg_epoch_loss:.3f}")

    # Log the completion of the QAT training process
    print('\nQAT Training finished.')
    
    # Return the model following the fine-tuning process
    return model_to_train



def evaluate_qat(model_to_eval, dataloader):
    """
    Evaluates the performance of a quantization-aware trained model on a dataset.

    Args:
        model_to_eval (torch.nn.Module): The model currently in a quantized state for evaluation.
        dataloader (torch.utils.data.DataLoader): The data provider for the evaluation process.
        device (str): The hardware device where the evaluation is performed.

    Returns:
        accuracy (float): The calculated accuracy of the model as a percentage.
    """

    # Explicitly define the device as CPU to ensure all operations run it, as they should be.
    device="cpu"
    
    # Transfer the model parameters to the target hardware device
    model_to_eval.to(device)
    
    # Switch the model to evaluation mode to disable layers like dropout
    model_to_eval.eval() 
    
    # Initialize counters for the number of correct predictions and total samples
    correct = 0
    total = 0
    
    # Deactivate gradient calculations to conserve memory and computational resources
    with torch.no_grad():
        # Initialize a progress bar to track evaluation across batches
        progress_bar = tqdm(dataloader, desc="Evaluating QAT Model")
        
        # Iterate through images and ground truth labels in the dataset
        for images, labels in progress_bar:

            # Move the batch of images and labels to the active device
            images, labels = images.to(device), labels.to(device)
            
            # Pass the input images through the model to obtain predictions
            outputs = model_to_eval(images)
            
            # Identify the class index with the highest activation value
            _, predicted = torch.max(outputs.data, 1)
            
            # Increment the total count of processed samples
            total += labels.size(0)
            
            # Increment the count of predictions that match the labels
            correct += (predicted == labels).sum().item()

    # Calculate the final accuracy percentage
    accuracy = 100 * correct / total
    
    # Log the resulting accuracy to the console
    print(f'Accuracy of the QAT model on the test set: {accuracy:.2f}%')

    # Return the resulting accuracy value
    return accuracy



def get_model_size(model):
    """
    Calculates the size of a model's state_dict in megabytes (MB) using a temporary file.

    Args:
        model (torch.nn.Module): The model whose storage size is to be measured.

    Returns:
        size_mb (float): The size of the model weights in megabytes.
    """

    # Create a temporary file using the tempfile module
    with tempfile.NamedTemporaryFile(delete=False, suffix=".p") as temp_file:
        # Retrieve the path of the created temporary file
        temp_file_path = temp_file.name
        # Save the model's state_dict to the temporary file
        torch.save(model.state_dict(), temp_file_path)

    # Get the size of the saved file in bytes
    size_bytes = os.path.getsize(temp_file_path)

    # Convert the size from bytes to megabytes
    size_mb = size_bytes / (1024 * 1024)

    # Clean up by deleting the temporary file from the system
    os.remove(temp_file_path)

    # Return the calculated size
    return size_mb



def measure_average_inference_time_ms(model, input_shape=(1, 3, 32, 32), num_runs=100):
    """
    Measures the average inference time of a PyTorch model in milliseconds.
    
    This function forces execution on the CPU to ensure compatibility with 
    quantized models and provide a consistent baseline for latency.

    Args:
        model (torch.nn.Module): The neural network model to be benchmarked.
        input_shape (tuple): The shape of the dummy input tensor to use for timing.
        num_runs (int): The number of inference iterations to perform for averaging.

    Returns:
        avg_time (float): The average time taken for a single forward pass in milliseconds.
    """
    
    # Force the entire measurement to run on the CPU
    device = torch.device("cpu")
    
    # Move the model to the CPU
    model.to(device)
    
    # Create a random input tensor on the CPU
    input_tensor = torch.rand(input_shape).to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for the warm-up and measurement phases
    with torch.no_grad():
        # Execute warm-up runs to ensure system caches and operations are initialized
        for _ in range(10):
            model(input_tensor)

    # Initialize a list to store the duration of each run
    timings = []
    
    # Begin the measurement phase
    with torch.no_grad():
        # Iterate for the specified number of runs
        for _ in range(num_runs):
            # Record the start time of the forward pass
            start_time = time.time()
            
            # Execute a single inference pass
            model(input_tensor)
            
            # Record the end time of the forward pass
            end_time = time.time()
            
            # Calculate duration and convert seconds to milliseconds
            timings.append((end_time - start_time) * 1000)

    # Calculate the average inference time across all measured runs
    avg_time = sum(timings) / len(timings)
    
    # Return the resulting average time
    return avg_time



def comparison_table(baseline_model_size, baseline_model_time, quantized_model_size, quantized_model_time, quantization_type):
    """
    Displays a model comparison table in a Jupyter-like environment.

    Args:
        baseline_model_size (float): The size of the baseline model in megabytes.
        baseline_model_time (float): The average inference latency of the baseline model in milliseconds.
        quantized_model_size (float): The size of the quantized model in megabytes.
        quantized_model_time (float): The average inference latency of the quantized model in milliseconds.
        quantization_type (str): The method of quantization used (e.g., 'Static', 'Dynamic', 'QAT').

    Returns:
        None: This function directly renders a Markdown table to the display output.
    """
    # Calculate the reduction in model size
    size_diff = baseline_model_size - quantized_model_size
    
    # Calculate the reduction in inference latency
    time_diff = baseline_model_time - quantized_model_time

    # Dynamically create the header for the quantized model column based on the technique used
    quantized_header = f"Quantized Model ({quantization_type})"

    # Construct a formatted Markdown table string to compare baseline and quantized metrics
    markdown_table_string = f"""
| | Baseline Model | {quantized_header} | Change |
|:---|:---|:---|:---|
| **Model Size (MB)** | {baseline_model_size:.2f} | {quantized_model_size:.2f} | {size_diff:.2f} |
| **Inference Latency (ms)** | {baseline_model_time:.2f} | {quantized_model_time:.2f} | {time_diff:.2f} |
"""

    # Directly render and display the Markdown table in the output cell
    display(Markdown(markdown_table_string))



def display_full_comparison(stats_dict):
    """
    Displays a markdown table comparing the size and inference time of all models.

    Args:
        stats_dict (dict): A dictionary where keys are model names (str) and 
                           values are tuples containing the model size (float) 
                           and inference time (float).

    Returns:
        None: This function renders a Markdown table directly to the display output.
    """
    
    # Start building the markdown table string with headers
    table = "| Model Type | Size (MB) | Inference Time (ms) |\n"
    
    # Add the separator row for the markdown table
    table += "|--------------------------|-----------|-----------------------|\n"
    
    # Iterate through the dictionary to add a row for each model entry
    for model_name, (size, time) in stats_dict.items():
        # Format the model name, size, and time into a table row string
        table += f"| {model_name:<24} | {size:<9.2f} | {time:<21.2f} |\n"

    # Render and display the final markdown table string
    display(Markdown(table))



def get_blip_vqa_model_and_processor():
    """
    Ensures the BLIP VQA model is available locally and loads the model and processor.

    Args:

    Returns:
        model (BlipForQuestionAnswering): The loaded BLIP model for question answering tasks.
        processor (BlipProcessor): The associated processor for image and text inputs.
    """
    # Define the model identifier from the Hugging Face Hub
    model_id = "Salesforce/blip-vqa-base"
    
    # Define the local directory where the model will be stored and loaded from
    local_path = "./blip-vqa-base-local"

    # Check if the model directory does not exist on the local filesystem
    if not os.path.exists(local_path):
        # Notify that the model is being downloaded to the local path
        print(f"Local model not found. Downloading '{model_id}' to local path: '{local_path}'")
        
        # Create the target directory and handle existing directory cases
        os.makedirs(local_path, exist_ok=True)
        
        try:
            # Download the pre-trained model weights from the hub
            temp_model = BlipForQuestionAnswering.from_pretrained(model_id)
            
            # Download the processor using the fast tokenizer implementation
            temp_processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
            
            # Save the downloaded model weights to the local directory
            temp_model.save_pretrained(local_path)
            
            # Save the processor configuration and files to the local directory
            temp_processor.save_pretrained(local_path)
            
            # Confirm successful download and save operation
            print("Model and processor downloaded and saved successfully.")
        except Exception as e:
            # Handle potential network or filesystem errors during the download process
            print(f"An error occurred during download: {e}")
            return None, None
    else:
        # Confirm that the local model directory was found
        print(f"Model already exists at local path: '{local_path}'.")

    try:
        # Indicate that the loading process from disk has started
        print(f"Loading model and processor from local path: {local_path}")
        
        # Load the model weights and configuration from the local filesystem
        model = BlipForQuestionAnswering.from_pretrained(local_path)
        
        # Load the processor and its tokenizer from the local filesystem
        processor = BlipProcessor.from_pretrained(local_path, use_fast=True)
        
        # Confirm that the objects are successfully initialized in memory
        print("Model and processor loaded successfully.")
        
        # Return the initialized model and processor
        return model, processor
    except Exception as e:
        # Handle errors during the loading process from local storage
        print(f"Failed to load model from local path: {e}")
        return None, None



def upload_jpg_widget():
    """
    Initializes and displays an interactive file upload widget for JPG images.

    Args:

    Returns:
        None: The function renders the widget and output area directly in the interface.
    """
    # Define the target directory for uploaded images
    output_image_folder = "./images"
    
    # Ensure the target directory exists, create it if it doesn't
    os.makedirs(output_image_folder, exist_ok=True)

    # Create the file upload widget specifically for JPG files
    uploader = widgets.FileUpload(
        accept='.jpg',  
        multiple=False, 
        description='Upload JPG (Max 5MB)' 
    )

    # Create an output widget to display status messages and results
    output_area = widgets.Output()

    def on_file_uploaded(change):
        """
        Processes the uploaded file and validates format, size, and storage path.

        Args:
            change (dict): Information about the state change in the upload widget.

        Returns:
            None: This internal function updates the display and saves files to disk.
        """
        
        # Get the new value from the change event object
        current_uploaded_value_tuple = change['new']

        # If the new value is empty, exit the function
        if not current_uploaded_value_tuple:
            return

        # Redirect display outputs to the designated output area
        with output_area:
            # Clear messages from any previous upload attempt
            output_area.clear_output() 

            # Extract the file data dictionary from the tuple
            file_data_dict = current_uploaded_value_tuple[0]
            
            # Retrieve the name of the uploaded file
            filename = file_data_dict['name']
            
            # Retrieve the file content as a byte stream
            file_content = file_data_dict['content'] 

            # Check if the file extension is specifically .jpg
            if not filename.lower().endswith('.jpg'):
                # Prepare a red error message for invalid formats
                error_msg_format = (
                    f"<p style='color:red;'>Error: Please upload a file with a ‘.jpg’ format. "
                    f"You uploaded: '{filename}'</p>"
                )
                # Render the error message
                display(HTML(error_msg_format))
                
                # Clear the invalid upload from the widget state
                uploader.value = () 
                return

            # Determine the file size in bytes
            file_size_bytes = len(file_content)
            
            # Define the maximum allowed size of 5 MB
            max_size_bytes = 5 * 1024 * 1024  

            # Validate that the file size is within the allowed limit
            if file_size_bytes > max_size_bytes:
                # Calculate the file size in megabytes for the message
                file_size_mb = file_size_bytes / (1024 * 1024)
                
                # Prepare a red error message for oversized files
                error_msg_size = (
                    f"<p style='color:red;'>Error: File '{filename}' is too large ({file_size_mb:.2f} MB). "
                    f"Please upload a file less than or equal to 5 MB.</p>"
                )
                # Render the error message
                display(HTML(error_msg_size))
                
                # Clear the oversized upload from the widget state
                uploader.value = () 
                return

            try:
                # Construct the full filesystem path to save the file
                save_path = os.path.join(output_image_folder, filename)

                # Open the file and write the byte content to disk
                with open(save_path, 'wb') as f:
                    f.write(file_content)

                # Format the file path for display in Python code format
                python_code_path = repr(save_path)

                # Construct a green success message with the final file path
                success_message = f"""
                <p style='color:green;'>File successfully uploaded!</p>
                <p>Please use the path as <code>image_path = {python_code_path}</code></p>
                """
                # Render the success message
                display(HTML(success_message))

            except Exception as e:
                # Handle potential filesystem errors during the write operation
                error_msg_save = f"<p style='color:red;'>Error saving file '{filename}': {e}</p>"
                
                # Render the specific saving error
                display(HTML(error_msg_save))
            finally:
                # Reset the uploader widget to its initial empty state
                uploader.value = ()

    # Monitor the uploader value and trigger processing on change
    uploader.observe(on_file_uploaded, names='value')

    # Render the uploader widget in the interface
    display(uploader)
    
    # Render the output area to display feedback below the widget
    display(output_area)



def perform_vqa(model, processor, image_path, question_text):
    """
    Performs Visual Question Answering by processing an image and a text query.

    Args:
        model (torch.nn.Module): The pre-trained VQA model used for generation.
        processor (transformers.ProcessorMixin): The processor for image and text normalization.
        image_path (str): The local filesystem path to the input image.
        question_text (str): The natural language question regarding the image.

    Returns:
        answer (str): The text-based answer generated by the model.
        inference_time (float): The duration of the model generation process in seconds.
    """
    # Open the image file from the provided path and convert it to RGB format
    raw_image = Image.open(image_path).convert('RGB')
    
    # Prepare the model inputs by encoding the image and the question text
    inputs = processor(raw_image, question_text, return_tensors="pt")
    
    # Record the high-resolution start time before the model forward pass
    start_time = time.perf_counter()
    
    # Generate the output tokens from the model based on the processed inputs
    output = model.generate(**inputs)
    
    # Record the high-resolution end time after the generation is complete
    end_time = time.perf_counter()
    
    # Calculate the elapsed time specifically for the generation step
    inference_time = end_time - start_time
    
    # Decode the generated output tokens into a human-readable string
    answer = processor.decode(output[0], skip_special_tokens=True)
    
    # Return the generated text answer and the measured inference time
    return answer, inference_time



def blip_comparison_table(
    question,
    baseline_answer,
    quantized_answer,
    baseline_size,
    quantized_size,
    baseline_time_s,
    quantized_time_s
):
    """
    Displays a formatted HTML table comparing the performance of the baseline
    and quantized BLIP model.

    Args:
        question (str): The text of the question asked to the model.
        baseline_answer (str): The answer generated by the original model.
        quantized_answer (str): The answer generated by the quantized model.
        baseline_size (float): The storage size of the baseline model in megabytes.
        quantized_size (float): The storage size of the quantized model in megabytes.
        baseline_time_s (float): The inference duration for the baseline model in seconds.
        quantized_time_s (float): The inference duration for the quantized model in seconds.

    Returns:
        None: This function renders an HTML table directly to the display output.
    """
    # Calculate the absolute difference in model size
    size_change = baseline_size - quantized_size
    
    # Initialize a default text for size change if calculation is not possible
    size_change_text = "N/A"
    
    # Check if baseline size is valid to calculate percentage changes
    if baseline_size > 0:
        # Calculate the percentage change in model size
        size_percent_change = (size_change / baseline_size) * 100
        
        # Determine if the change represents a reduction or an increase
        size_change_label = "reduction" if size_change >= 0 else "increase"
        
        # Format the size change string with absolute and percentage values
        size_change_text = f"{size_change:.2f} ({abs(size_percent_change):.1f}% {size_change_label})"

    # Calculate the absolute difference in inference time
    time_change_s = baseline_time_s - quantized_time_s
    
    # Initialize a default text for time change if calculation is not possible
    time_change_text = "N/A"
    
    # Check if baseline time is valid to calculate percentage changes
    if baseline_time_s > 0:
        # Calculate the percentage change in inference latency
        time_percent_change = (time_change_s / baseline_time_s) * 100
        
        # Determine if the change represents a reduction or an increase in speed
        time_change_label = "reduction" if time_change_s >= 0 else "increase"
        
        # Format the time change string with absolute and percentage values
        time_change_text = f"{time_change_s:.4f} ({abs(time_percent_change):.1f}% {time_change_label})"
        
    # Construct the HTML template for the comparison table
    html_string = f"""
    <p><b>Question:</b> {question}</p>
    <table border="1" style="width:100%; border-collapse: collapse; text-align: left;">
      <tr style="background-color: #4A5568; color: white;">
        <th style="padding: 8px;"></th>
        <th style="padding: 8px;">Model Size (MB)</th>
        <th style="padding: 8px;">Inference Time (s)</th>
        <th style="padding: 8px;">Answer</th>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Baseline Model</b></td>
        <td style="padding: 8px;">{baseline_size:.2f}</td>
        <td style="padding: 8px;">{baseline_time_s:.4f}</td>
        <td style="padding: 8px;">{baseline_answer}</td>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Quantized Model (Dynamic)</b></td>
        <td style="padding: 8px;">{quantized_size:.2f}</td>
        <td style="padding: 8px;">{quantized_time_s:.4f}</td>
        <td style="padding: 8px;">{quantized_answer}</td>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Change</b></td>
        <td style="padding: 8px;"><b>{size_change_text}</b></td>
        <td style="padding: 8px;"><b>{time_change_text}</b></td>
        <td style="padding: 8px;">---</td>
      </tr>
    </table>
    """

    # Render and display the generated HTML content in the notebook environment
    display(HTML(html_string))


