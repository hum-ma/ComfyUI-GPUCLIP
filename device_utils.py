import torch
import logging
import hashlib
import psutil
import comfy.model_management as mm
import gc

logger = logging.getLogger("GPUClip")

_DEVICE_LIST_CACHE = None

def get_device_list():
    """
    Enumerate ALL physically available devices that can store torch tensors.
    This includes all device types supported by ComfyUI core.
    Results are cached after first call since devices don't change during runtime.
    
    Returns a comprehensive list of all available devices across all types:
    - CPU (always available)
    - CUDA devices (NVIDIA GPUs)
    - XPU devices (Intel GPUs)
    - NPU devices (Ascend NPUs from Huawei)
    - MLU devices (Cambricon MLUs)
    - MPS device (Apple Metal)
    - DirectML devices (Windows DirectML)
    - CoreX/IXUCA devices
    """
    global _DEVICE_LIST_CACHE
    
    if _DEVICE_LIST_CACHE is not None:
        return _DEVICE_LIST_CACHE
    
    devs = []
    
    devs.append("cpu")
    
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devs += [f"cuda:{i}" for i in range(device_count)]
        logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} CUDA device(s)")
    
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass
    
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        devs += [f"xpu:{i}" for i in range(device_count)]
        logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} XPU device(s)")
    
    try:
        import torch_npu
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            device_count = torch.npu.device_count()
            devs += [f"npu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} NPU device(s)")
    except ImportError:
        pass
    
    try:
        import torch_mlu
        if hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
            device_count = torch.mlu.device_count()
            devs += [f"mlu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} MLU device(s)")
    except ImportError:
        pass
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
        logger.debug("[MultiGPU_Device_Utils] Found MPS device")
    
    try:
        import torch_directml
        adapter_count = torch_directml.device_count()
        if adapter_count > 0:
            devs += [f"directml:{i}" for i in range(adapter_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {adapter_count} DirectML adapter(s)")
    except ImportError:
        pass
    
    try:
        if hasattr(torch, "corex"):
            if hasattr(torch.corex, "device_count"):
                device_count = torch.corex.device_count()
                devs += [f"corex:{i}" for i in range(device_count)]
                logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} CoreX device(s)")
            else:
                devs.append("corex:0")
                logger.debug("[MultiGPU_Device_Utils] Found CoreX device")
    except ImportError:
        pass
    
    _DEVICE_LIST_CACHE = devs
    
    logger.debug(f"[MultiGPU_Device_Utils] Device list initialized: {devs}")
    
    return devs

def is_accelerator_available():
    """Check if any GPU or accelerator device is available including CUDA, XPU, NPU, MLU, MPS, DirectML, or CoreX."""
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return True
    
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        return True
    
    try:
        import torch_npu
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            return True
    except ImportError:
        pass

    try:
        import torch_mlu
        if hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
            return True
    except ImportError:
        pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True

    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            return True
    except ImportError:
        pass

    if hasattr(torch, "corex"):
        return True
    
    return False

def is_device_compatible(device_string):
    """Check if a device string represents a valid available device."""
    available_devices = get_device_list()
    return device_string in available_devices

def get_device_type(device_string):
    """Extract device type from device string (e.g. 'cuda' from 'cuda:0')."""
    if ":" in device_string:
        return device_string.split(":")[0]
    return device_string

def parse_device_string(device_string):
    """Parse device string into (device_type, device_index) tuple."""
    if ":" in device_string:
        parts = device_string.split(":")
        return parts[0], int(parts[1])
    return device_string, None

