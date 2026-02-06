"""
ComfyUI-GPUClip Wrapper Functions
All node override/wrapper generation functions consolidated in one location
"""

import copy
from .device_utils import get_device_list


# ============================================================================
# STANDARD MULTIGPU WRAPPERS (Device selection without DisTorch)
# ============================================================================

def override_class(cls):
    """Standard MultiGPU device override for UNet/VAE models"""
    from . import set_current_device
    
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "GPUCLIP"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            if device is not None:
                set_current_device(device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            return out

    return NodeOverride


def override_class_clip(cls):
    """Standard MultiGPU device override for CLIP models (with device kwarg workaround)"""
    from . import set_current_text_encoder_device
    
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "GPUCLIP"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            if device is not None:
                set_current_text_encoder_device(device)
            kwargs['device'] = 'default'
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            return out

    return NodeOverride


def override_class_clip_no_device(cls):
    """Standard MultiGPU device override for Triple/Quad CLIP models (no device kwarg workaround)"""
    from . import set_current_text_encoder_device
    
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "GPUCLIP"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            if device is not None:
                set_current_text_encoder_device(device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            return out

    return NodeOverride
