import json

def load_benchmark_data(json_data):
    """
    Load benchmark data from a JSON string, dictionary, or file path
    
    Args:
        json_data: JSON string, dictionary, or file path
        
    Returns:
        tuple: (results, metadata, model_name)
    """
    if isinstance(json_data, str):
        try:
            # Try to parse as JSON string
            data = json.loads(json_data)
        except json.JSONDecodeError:
            # If not a valid JSON string, assume it's a file path
            with open(json_data, 'r') as f:
                data = json.load(f)
    else:
        # Already a dictionary
        data = json_data
    
    # Extract the results
    results = data.get("results", {})
    metadata = data.get("metadata", {})
    
    # Get model information if available
    model_name = metadata.get("model", "Unknown Model")
    
    return results, metadata, model_name