import yaml
import os


def join_constructor(loader, node):
    """
    Joins the list of strings into a proper filesystem path.
    Uses os.path.join to ensure correct path separators for the current OS.
    """
    seq = loader.construct_sequence(node)
    
    # The first element is the base path
    if not seq:
        return ""
        
    base_path = seq[0]
    
    # Join remaining elements as path components
    if len(seq) > 1:
        # Remove any leading slashes from path components to avoid
        # os.path.join treating them as absolute paths
        components = [comp.lstrip("/").lstrip("\\") for comp in seq[1:]]
        return os.path.join(base_path, *components)
    
    return base_path


def env_var_or_default_constructor(loader, node):
    """
    Get value from environment variable or use default.
    Usage in YAML: !env_var_or_default [ENV_VAR_NAME, default_value]
    """
    seq = loader.construct_sequence(node)
    if len(seq) != 2:
        raise ValueError("!env_var_or_default requires exactly 2 args: [ENV_VAR_NAME, default_value]")
    
    env_var_name, default_value = seq
    return os.environ.get(env_var_name, default_value)


# Register the constructors with the SafeLoader
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
yaml.add_constructor("!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader)
