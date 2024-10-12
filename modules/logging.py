import torch
from torch import nn
import functools
import inspect

def log_io(func):
    """
    Decorator to log inputs and outputs of nn.Module methods.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function with logging functionality.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip logging if disabled or if the function is in the disabled list
        if not self.logging_enabled or func.__name__ in self.disabled_logging_functions:
            return func(self, *args, **kwargs)

        def log_item(item, name, level=0, is_root=False):
            indent = "    " * level
            if isinstance(item, torch.Tensor):
                min_val, max_val = item.min().item(), item.max().item()
                print(f"{indent}Tensor '{name}' shape: {item.shape}, dtype: {item.dtype}, "
                      f"device: {item.device}, min/max: {min_val:.3f}/{max_val:.3f}")
                if self.print_full_tensors:
                    print(f"{indent}Full tensor content:\n{item}")
            elif isinstance(item, tuple):
                if is_root and level == 0:
                    for idx, sub_item in enumerate(item):
                        log_item(sub_item, f"{name}[{idx}]", level)
                else:
                    print(f"{indent}Tuple '{name}':")
                    for idx, sub_item in enumerate(item):
                        log_item(sub_item, f"{name}[{idx}]", level + 1)
            elif isinstance(item, list):
                print(f"{indent}List '{name}':")
                for idx, sub_item in enumerate(item):
                    if isinstance(sub_item, torch.Tensor):
                        min_val, max_val = sub_item.min().item(), sub_item.max().item()
                        print(f"{indent}    Tensor '{name}[{idx}]' shape: {sub_item.shape}, dtype: {sub_item.dtype}, "
                              f"device: {sub_item.device}, min/max: {min_val:.3f}/{max_val:.3f}")
                        if self.print_full_tensors:
                            print(f"{indent}    Full tensor content:\n{sub_item}")
                    else:
                        log_item(sub_item, f"{name}[{idx}]", level + 1)
            elif isinstance(item, dict):
                print(f"{indent}Dict '{name}':")
                for key, sub_item in item.items():
                    log_item(sub_item, f"{name}[{key}]", level + 1)
            else:
                print(f"{indent}{type(item).__name__} '{name}': Value={item}")
        
        # Bind arguments to their names
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Logging inputs
        print(f"\n{'='*20}Entering {self.__class__.__name__}.{func.__name__}{'='*20}")
        print("Inputs:")
        for name, value in bound_args.arguments.items():
            if name != 'self':
                log_item(value, name)

        # Execute the original function
        result = func(self, *args, **kwargs)
        
        # Logging outputs
        print("\nOutputs:")
        if isinstance(result, tuple):
            log_item(result, "output", is_root=True)
        else:
            log_item(result, "output")

        print(f"{'='*20}Exiting {self.__class__.__name__}.{func.__name__}{'='*20}")
        return result
    return wrapper

class LoggingModule(nn.Module):
    """
    A PyTorch nn.Module with logging capabilities for educational purposes.

    Attributes:
        logging_enabled (bool): Flag to enable or disable logging.
        disabled_logging_functions (set): Set of function names for which logging is disabled.
        print_full_tensors (bool): Flag to enable printing of full tensor contents.
    """
    def __init__(self):
        super().__init__()
        self.logging_enabled = False
        self.disabled_logging_functions = set()
        self.print_full_tensors = False

    def enable_logging(self):
        """Enable logging for all methods."""
        self.logging_enabled = True

    def disable_logging(self):
        """Disable logging for all methods."""
        self.logging_enabled = False

    def disable_function_logging(self, func_name):
        """
        Disable logging for a specific function.

        Args:
            func_name (str): Name of the function to disable logging for.
        """
        self.disabled_logging_functions.add(func_name)

    def enable_function_logging(self, func_name):
        """
        Enable logging for a specific function.

        Args:
            func_name (str): Name of the function to enable logging for.
        """
        self.disabled_logging_functions.discard(func_name)

    def enable_full_tensor_printing(self):
        """Enable printing of full tensor contents."""
        self.print_full_tensors = True

    def disable_full_tensor_printing(self):
        """Disable printing of full tensor contents."""
        self.print_full_tensors = False