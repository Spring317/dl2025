def read_txt_file(file_path):
    """
    Reads a text file and returns its content as a list of lines.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        list: A list of lines from the text file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]  # Remove leading/trailing whitespace

def read_config(config_path):
    """
    Reads a configuration file and returns a dictionary of parameter values.
    
    Args:
        config_path (str): The path to the configuration file.
        
    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    config = {}
    lines = read_txt_file(config_path)
    
    for line in lines:
        # Skip empty lines and comments
        if not line or line.startswith('//'):
            continue
            
        # Split the line into key and value
        key, value = line.split('=')
        
        # Try to convert to integer if possible
        try:
            config[key] = int(value)
        except ValueError:
            # If not an integer, keep as string
            config[key] = value
            
    return config