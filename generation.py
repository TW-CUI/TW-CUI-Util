from datetime import timezone, datetime
import hashlib
import random

import torch

import comfy.model_management
import comfy.sd
import comfy.samplers
import comfy.utils

import folder_paths
from nodes import MAX_RESOLUTION

from ._base import TWCUI_Util_BaseNode as BaseNode, GLOBAL_CATEGORY
from ._static import vae_list, load_taesd

MODULE_CATEGORY = f"{GLOBAL_CATEGORY}/generation"


# Initialize the random system anew. This is because some extensions may alter
# this seed generation process and cause problems.
initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
seed_random_state = random.getstate()
random.setstate(initial_random_state)


def _new_random_seed():
    """ Gets a new random seed from the seed_random_state and resetting the previous state."""
    global seed_random_state
    prev_random_state = random.getstate()
    random.setstate(seed_random_state)
    seed = random.randint(1, 18446744073709551615)
    seed_random_state = random.getstate()
    random.setstate(prev_random_state)
    return seed


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
     - Seed
     - control_after_generate (defines seed behavior)

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
     - seed (INT)
    """

    def __init__(self):
        super().__init__()
        self.device = comfy.model_management.intermediate_device()

    CATEGORY = MODULE_CATEGORY

    @staticmethod
    def _load_checkpoint(ckpt_name, output_vae=True, output_clip=True) -> tuple[comfy.model_patcher.ModelPatcher,
                                                                                comfy.sd.CLIP, comfy.sd.VAE]:
        # Implementation taken from CheckpointLoaderSimple in ComfyUI nodes.py
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
                "cfg": ("FLOAT", {
                    "default": 8.00,
                    "min": 0.00,
                    "max": 20.00,
                    "step": 0.25,
                    "display": "number"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "CLIP", "VAE", "STRING", "LATENT", "INT", "INT", "INT", "FLOAT",
                    comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("MODEL", "model_hash", "CLIP", "VAE", "vae_hash", "LATENT", "width", "height", "steps", "cfg",
                    "sampler_name", "scheduler", "seed")

    def process(self, ckpt_name: str, vae_name: str, image_width: int, image_height: int, sampling_steps: int,
                cfg: float, sampler_name: str, scheduler_name: str, seed: int) -> tuple:
        MODEL, CLIP, VAE0 = self._load_checkpoint(ckpt_name)
        model_sha256_hash = hashlib.sha256()
        with open(folder_paths.get_full_path("checkpoints", ckpt_name), "rb") as f:
            # Read the file in chunks to avoid loading the entire file into memory
            for byte_block in iter(lambda: f.read(4096), b""):
                model_sha256_hash.update(byte_block)
        model_hash = model_sha256_hash.hexdigest()[:10]

        if vae_name in ["taesd", "taesdxl"]:
            sd = load_taesd(vae_name)
            vae_hash = "unknown (taesd or taesdxl VAE)"
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae_sha256_hash = hashlib.sha256()
            with open(vae_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    vae_sha256_hash.update(byte_block)
            vae_hash = vae_sha256_hash.hexdigest()[:10]
            sd = comfy.utils.load_torch_file(vae_path)
        VAE = comfy.sd.VAE(sd=sd)

        if seed == -1:
            # When seed value is -1, we generate a random value.
            original_seed = seed
            seed = new_random_seed()

        batch_size = 1
        latent = torch.zeros([batch_size, 4, image_height // 8, image_width // 8], device=self.device)
        LATENT = {"samples": latent}

        return (MODEL, model_hash, CLIP, VAE, vae_hash, LATENT, image_width, image_height, sampling_steps, cfg,
                sampler_name, scheduler_name, seed)


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


class TWCUI_Util_GenerationPrompts(BaseNode):
    """
    This is a multi-field TEXT node that allows entering a positive and negative
    prompt and pass them both out.

    Contains two multiline text input fields, neg_prompt is optional.

    Produces the PROMPT and NEGPROMPT as STRING, also produces POSITIVE and NEGATIVE CONDITIONING.
    """

    RETURN_TYPES = ("STRING", "STRING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("prompt", "neg_prompt", "POSITIVE", "NEGATIVE")

    CATEGORY = MODULE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "CLIP": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True})
            },
            "optional": {
                "neg_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True})
            }
        }

    @staticmethod
    def _encode(clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def process(self, CLIP: comfy.sd.CLIP, prompt: str, neg_prompt: str) -> tuple[str, str, object, object]:
        positive = self._encode(CLIP, prompt)
        negative = self._encode(CLIP, neg_prompt)

        return prompt, neg_prompt, positive, negative
