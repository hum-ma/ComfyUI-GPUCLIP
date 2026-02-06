import torch
import folder_paths
from nodes import NODE_CLASS_MAPPINGS
from .device_utils import get_device_list

class GPUClipDeviceSelector:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        return {
            "required": {
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0]})
            }
        }

    RETURN_TYPES = (get_device_list(),)
    RETURN_NAMES = ("device",)
    FUNCTION = "select_device"
    CATEGORY = "GPUCLIP"

    def select_device(self, device):
        """Select target device from available device list."""
        return (device,)

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        import nodes
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg"
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        """Get combined list of CLIP and CLIP_GGUF model files."""
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        """Load CLIP model data from checkpoint paths."""
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_data(ckpt_paths)

    def load_patcher(self, clip_paths, clip_type, clip_data):
        """Create ModelPatcher for CLIP model."""
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_patcher(clip_paths, clip_type, clip_data)

    def load_clip(self, clip_name, type="stable_diffusion", device=None):
        """Load CLIP model from GGUF or standard format."""
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_clip(clip_name, type)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        import nodes
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": base["required"]["type"],
            }
        }

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type, device=None):
        """Load dual CLIP model configuration."""
        original_loader = NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
        clip = original_loader.load_clip(clip_name1, clip_name2, type)
        clip[0].patcher.load(force_patch_weights=True)
        return clip


class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "clip_name3": file_options,
            }
        }

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        """Load triple CLIP model configuration for SD3."""
        original_loader = NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUF"]()
        return original_loader.load_clip(clip_name1, clip_name2, clip_name3, type)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "clip_name4": file_options,
        }
    }

    TITLE = "QuadrupleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        """Load quadruple CLIP model configuration."""
        original_loader = NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderGGUF"]()
        return original_loader.load_clip(clip_name1, clip_name2, clip_name3, clip_name4, type)

