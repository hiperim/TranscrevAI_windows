import os
import logging

def validate_audio_format(func):
    def wrapper(file_path):
        if not (file_path.endswith(".wav") or file_path.endswith(".mp4")):
            raise TypeError(f"Unsupported audio format: {file_path}")
        return func(file_path)
    return wrapper

def setup_app_logging():
    """
    Sets up basic logging for the app
    Creates a log file that saves important app information
    """
    # Create the logs folder if it doesn't exist
    logs_folder = "logs"
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    
    # Path for our log file
    log_file = os.path.join(logs_folder, "app.log")
    
    # Remove any existing logging setup
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup new logging configuration
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console output for logs too
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger(__name__)