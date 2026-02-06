import torch
import logging
import os
import folder_paths
import comfy.model_management as mm
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
from .device_utils import (
    get_device_list,
    is_accelerator_available,
)

MGPU_MM_LOG = False
DEBUG_LOG = False

logger = logging.getLogger("GPUClip")
logger.propagate = False

if not logger.handlers:
    log_level = logging.DEBUG if DEBUG_LOG else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

def check_module_exists(module_path):
    """Check if a custom node module exists in ComfyUI custom_nodes directory."""
    full_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], module_path)
    logger.debug(f"[GPUClip] Checking for module at {full_path}")
    if not os.path.exists(full_path):
        logger.debug(f"[GPUClip] Module {module_path} not found - skipping")
        return False
    logger.debug(f"[GPUClip] Found {module_path}, creating compatible MultiGPU nodes")
    return True

current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()

def set_current_device(device):
    """Set the current device context for MultiGPU operations."""
    global current_device
    current_device = device
    logger.debug(f"[GPUClip Initialization] current_device set to: {device}")

def set_current_text_encoder_device(device):
    """Set the current text encoder device context for CLIP models."""
    global current_text_encoder_device
    current_text_encoder_device = device
    logger.debug(f"[GPUClip Initialization] current_text_encoder_device set to: {device}")

def get_torch_device_patched():
    """Return MultiGPU-aware device selection for patched mm.get_torch_device."""
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_device) if str(current_device) in devs else torch.device("cpu")
    logger.debug(f"[GPUClip Core Patching] get_torch_device_patched returning device: {device} (current_device={current_device})")
    return device

def text_encoder_device_patched():
    """Return MultiGPU-aware text encoder device for patched mm.text_encoder_device."""
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_text_encoder_device) if str(current_text_encoder_device) in devs else torch.device("cpu")
    logger.info(f"[GPUClip Core Patching] text_encoder_device_patched returning device: {device} (current_text_encoder_device={current_text_encoder_device})")
    return device

mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched

from .nodes import (
    GPUClipDeviceSelector,
    CLIPLoaderGGUF,
    DualCLIPLoaderGGUF,
    TripleCLIPLoaderGGUF,
    QuadrupleCLIPLoaderGGUF,
)

from .wrappers import (
    override_class,
    override_class_clip,
    override_class_clip_no_device,
)

NODE_CLASS_MAPPINGS = {
    "GPUClipDeviceSelector": GPUClipDeviceSelector,
}

NODE_CLASS_MAPPINGS["GPUCLIPLoader"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["GPUDualCLIPLoader"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
NODE_CLASS_MAPPINGS["GPUTripleCLIPLoader"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
NODE_CLASS_MAPPINGS["GPUQuadrupleCLIPLoader"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])

def register_and_count(module_names, node_map):
    """Register MultiGPU node wrappers for detected custom node modules."""
    found = False
    for name in module_names:
        if check_module_exists(name):
            found = True
            break
    
    count = 0
    if found:
        initial_len = len(NODE_CLASS_MAPPINGS)
        for key, value in node_map.items():
            NODE_CLASS_MAPPINGS[key] = value
        count = len(NODE_CLASS_MAPPINGS) - initial_len
        
    return found

gguf_nodes = {
    "GPUCLIPLoaderGGUF": override_class_clip(CLIPLoaderGGUF),
    "GPUDualCLIPLoaderGGUF": override_class_clip(DualCLIPLoaderGGUF),
    "GPUTripleCLIPLoaderGGUF": override_class_clip_no_device(TripleCLIPLoaderGGUF),
    "GPUQuadrupleCLIPLoaderGGUF": override_class_clip_no_device(QuadrupleCLIPLoaderGGUF)
}
register_and_count(["ComfyUI-GGUF", "comfyui-gguf"], gguf_nodes)

logger.info(f"[GPUClip] mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
