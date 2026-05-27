import os
import subprocess
import time

import mlflow
from IPython.display import display, Markdown


def show_ui_navigation_instructions(display_instructions=False):
    """
    Displays instructions for navigating the MLflow UI as markdown.

    Args:
        display_instructions (bool): Set to True to display the instructions.
                                     Defaults to False.
    """
    if display_instructions:
        instructions = """
* Once in the MLflow UI, you will find your experiment name (e.g., `CIFAR10_CNN`) in the list on the left and click on it.

<p align="center">
  <img src="./nb_images/step_1.png" alt="Step 1 Image" style="width: 85%;"/>
</p>

* Depending on how many times you conducted the experiment, you may see a list of "Runs". Your latest run will be at the top; you will click it to open.

<p align="center">
  <img src="./nb_images/step_2.png" alt="Step 2 Image" style="width: 75%;"/>
</p>

* You will be able to explore information about your selected run using the different tabs.

<p align="center">
  <img src="./nb_images/step_3.png" alt="Step 3 Image" style="width: 60%;"/>
</p>

* You can then go to the `Artifacts` tab to see all logged files, like the saved `confusion_matrix.png`. You will be able to click to view them.

<p align="center">
  <img src="./nb_images/step_4.png" alt="Step 4 Image" style="width: 60%;"/>
</p>

### Terminating the MLflow UI Server

* When you are done, you can stop the server by running `mlflow_process.terminate()` in a notebook cell.
        """
        display(Markdown(instructions))



def start_mlflow_ui(base_path='/mlflow', port=5000, jupyter_port=8888):
    """
    Starts the MLflow UI server with nginx reverse proxy support.

    This function starts the MLflow UI on localhost with a static prefix for
    nginx reverse proxy integration and runs in the background without blocking.

    Args:
        base_path (str): The base path for the MLflow UI (nginx location). Defaults to '/mlflow'.
        port (int): The port MLflow will listen on. Defaults to 5000.
        jupyter_port (int): The Jupyter server port for the access URL. Defaults to 8888.

    Returns:
        subprocess.Popen: The MLflow server process object.
    """
    # Start MLflow UI in background with base path
    print("Starting MLflow UI...")
    process = subprocess.Popen(
        ['mlflow', 'ui', '--host', '0.0.0.0', '--port', str(port), '--static-prefix', base_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print("Waiting for server to start...")
    time.sleep(3)

    print("\n" + "="*60)
    print("MLflow UI is running in the background!")
    print("="*60)
    coursera_host = os.getenv('WORKSPACE_ID')
    if coursera_host:
        print(f"Access it at: http://{os.environ['WORKSPACE_ID']}.labs.coursera.org{base_path}")
    else:
        display(Markdown("Access from: [Open MLflow UI](/mlflow/)"))
    print("\nThe server is running in the background. You can continue with the notebook.")
    print("To stop the server later, run: mlflow_process.terminate()")
    print("="*60)

    return process
    

def display_mlflow_run_details(run_id):
    """
    Fetches and displays the details of a specific MLflow run in a formatted way.

    This includes the experiment info, parameters, final metric values, and a list
    of artifacts associated with the run.

    Args:
        run_id (str): The unique ID of the MLflow run to inspect.
    """
    # Validate the run_id input
    if not run_id or not run_id.strip():
        print("Error: A valid Run ID must be provided. It cannot be empty or just whitespace.")
        return

    client = mlflow.tracking.MlflowClient()
    print(f"MLflow Client Initialized. Fetching details for Run ID: {run_id}\n")

    # Define the expected parameters and metrics from the training script
    logged_params_keys = [
        "model_type", "optimizer", "initial_lr", 
        "scheduler", "batch_size", "random_seed"
    ]
    logged_metrics_keys = {
        "train_loss": " (logged per epoch)",
        "val_loss": " (logged per epoch)",
        "accuracy": " (logged per epoch)",
        "learning_rate": " (logged per epoch)",
        "best_accuracy": ""
    }

    try:
        run_info = client.get_run(run_id)
        print(f"Details for Run ID: {run_info.info.run_id}")
        experiment_id_of_run = run_info.info.experiment_id
        
        try:
            experiment_details = client.get_experiment(experiment_id_of_run)
            print(f"Belongs to Experiment: {experiment_details.name} (ID: {experiment_id_of_run})")
        except Exception as e:
            print(f"Belongs to Experiment ID: {experiment_id_of_run} (Could not fetch details: {e})")

        # Print Logged Parameters
        print("\n  Parameters:")
        found_params = False
        for key in logged_params_keys:
            if key in run_info.data.params:
                print(f"    - {key}: {run_info.data.params[key]}")
                found_params = True
        if not found_params:
            print("    No specific logged parameters found for this run matching the expected keys.")

        # Print Logged Metrics (final values)
        print("\n  Metrics (final values):")
        found_metrics = False
        for key, note in logged_metrics_keys.items():
            if key in run_info.data.metrics:
                metric_value = run_info.data.metrics[key]
                try:
                    print(f"    - {key}: {float(metric_value):.4f}{note}")
                except ValueError:
                    print(f"    - {key}: {metric_value}{note}")
                found_metrics = True
        if not found_metrics:
            print("    No specific logged metrics found for this run matching the expected keys.")

        # List Logged Artifacts
        print("\n  Artifacts:")
        artifacts = client.list_artifacts(run_info.info.run_id)
        if artifacts:
            for artifact in artifacts:
                if "best_model_checkpoint_epoch_" in artifact.path and artifact.path.endswith(".pt"):
                    print(f"    - Checkpoint '{artifact.path}' was saved.")
                elif artifact.path == "confusion_matrix.png":
                    print(f"    - Artifact '{artifact.path}' was saved.")
                elif artifact.path == "cifar10_cnn_model_final" and artifact.is_dir:
                    print(f"    - PyTorch Model '{artifact.path}' was saved (directory).")
                else:
                    status = "Directory" if artifact.is_dir else "File"
                    print(f"    - '{artifact.path}' ({status}).")
        else:
            print("    No artifacts found for this run.")
        print("  ---")

    except mlflow.exceptions.MlflowException as e:
        print(f"Error fetching run details for Run ID '{run_id}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
