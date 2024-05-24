import comfy.utils
import comfy.sd
import comfy.samplers

import folder_paths
from nodes import MAX_RESOLUTION

from ._base import TWCUI_Util_BaseNode as BaseNode, GLOBAL_CATEGORY
from ._static import vae_list, load_taesd

MODULE_CATEGORY = f"{GLOBAL_CATEGORY}/generation"


class TWCUI_Util_GenerationParameters(BaseNode):
    """
    A more complex node that helps to define a number of generation parameters at once
    for passing into other nodes.

    Primarily, it is used to specify the:
     - Checkpoint/Model
     - VAE
     - Image width
     - Image height
     - Sampling steps
     - CFG scale
     - Sampler
     - Scheduler

    Outputs:
     - MODEL
     - VAE
     - CLIP
     - width (INT)
     - height (INT)
     - steps (INT)
     - cfg_scale (FLOAT)
     - SAMPLER
     - SCHEDULER
    """

    CATEGORY = MODULE_CATEGORY

    @staticmethod
    def _load_checkpoint(ckpt_name, output_vae=True, output_clip=True) -> tuple[object, comfy.sd.CLIP, comfy.sd.VAE]:
        # Implementation takekn from CheckpointLoaderSimple in ComfyUI nodes.py
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae, output_clip,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (vae_list(),),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 8,
                    "max": MAX_RESOLUTION,
                    "step": 8,
                    "display": "number"
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 8,
                    "max": MAX_RESOLUTION,
                    "step": 8,
                    "display": "number"
                }),
                "sampling_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 8.00,
                    "min": 0.00,
                    "max": 10.00,
                    "step": 0.25,
                    "round": 2,
                    "display": "number"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,)
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "INT", "INT", "FLOAT",
                    comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "width", "height", "cfg_scale", "sampler_name", "scheduler_name")

    def process(self, ckpt_name: str, vae_name: str, image_width: int, image_height: int, sampling_steps: int,
                cfg_scale: float, sampler_name: str, scheduler_name: str) -> tuple:
        MODEL, CLIP, baseVAE = self._load_checkpoint(ckpt_name)
        print(f"{type(MODEL)}, {type(CLIP)}, {type(baseVAE)}")
        if vae_name in ["taesd", "taesdxl"]:
            sd = load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        VAE = comfy.sd.VAE(sd=sd)

        return MODEL, CLIP, VAE, image_width, image_height, sampling_steps, cfg_scale, sampler_name, scheduler_name


class TWCUI_Util_CommonSDXLResolutions(BaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "dimensions": ([
                   "640 x 1536 (Portrait)",
                   "768 x 1344 (Portrait)",
                   "832 x 1216 (Portrait)",
                   "896 x 1152 (Portrait)",
                   "1024 x 1024 (Square)",
                   "1152 x 896 (Landscape)",
                   "1216 x 832 (Landscape)",
                   "1344 x 768 (Landscape)",
                   "1536 x 640 (Landscape)"
                ],)
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height")

    CATEGORY = MODULE_CATEGORY

    def process(self, **kwargs) -> tuple[int, int]:
        dim, orient = dimensions.split(' (', 1)
        orient = orient.strip('()')
        dims: str = dim.lower().split(' x ')

        return int(dims[0]), int(dims[1])
