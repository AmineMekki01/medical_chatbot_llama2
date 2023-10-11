import yaml

def load_config(file_path):
    """
    Load the config from the file_path  
    parameters:
    -----------
        file_path (str): The path to the config file    
    Returns:
    --------
        dict: The config dictionary

    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

