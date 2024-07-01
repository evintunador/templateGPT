import torch
from torch import nn
import functools
import inspect

def log_io(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
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
            elif isinstance(item, bool):
                print(f"{indent}Bool '{name}': Value={item}")
            elif isinstance(item, int):
                print(f"{indent}Integer '{name}': Value={item}")
            elif isinstance(item, float):
                print(f"{indent}Float '{name}': Value={item}")
            else:
                print(f"{indent}Other-type '{name}': Type={type(item).__name__}, Value={item}")
        
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        print(f"\n{'='*20}Entering {self.__class__.__name__}.{func.__name__}{'='*20}")
        print("Inputs:")
        for name, value in bound_args.arguments.items():
            if name != 'self':
                log_item(value, name)

        result = func(self, *args, **kwargs)
        
        print("\nOutputs:")
        if isinstance(result, tuple):
            log_item(result, "output", is_root=True)
        else:
            log_item(result, "output")

        print(f"{'='*20}Exiting {self.__class__.__name__}.{func.__name__}{'='*20}")
        return result
    return wrapper

class LoggingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.logging_enabled = False
        self.disabled_logging_functions = set()
        self.print_full_tensors = False  # New attribute

    def enable_logging(self):
        self.logging_enabled = True

    def disable_logging(self):
        self.logging_enabled = False

    def disable_function_logging(self, func_name):
        self.disabled_logging_functions.add(func_name)

    def enable_function_logging(self, func_name):
        self.disabled_logging_functions.discard(func_name)

    def enable_full_tensor_printing(self):
        self.print_full_tensors = True

    def disable_full_tensor_printing(self):
        self.print_full_tensors = False