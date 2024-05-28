from datetime import datetime
import hashlib
import os.path
import json
import random

import torch

# noinspection PyUnresolvedReferences,PyPackageRequirements
import comfy.model_management
# noinspection PyUnresolvedReferences,PyPackageRequirements
import comfy.sd
# noinspection PyUnresolvedReferences,PyPackageRequirements
import comfy.samplers
# noinspection PyUnresolvedReferences,PyPackageRequirements
import comfy.utils

# noinspection PyUnresolvedReferences,PyPackageRequirements
import folder_paths
# noinspection PyUnresolvedReferences,PyPackageRequirements
from nodes import MAX_RESOLUTION

from ._base import TWCUI_Util_BaseNode as BaseNode, GLOBAL_CATEGORY

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
    seed = random.randint(1, 0xffffffffffffffff)
    seed_random_state = random.getstate()
    random.setstate(prev_random_state)
    return seed


class TWCUI_Util_GenerationParameters(BaseNode):
    """
    A more complex node that helps to define a number of generation parameters at once
    for passing into other nodes.

    Primarily, it is used to specify the:
     - Image width
     - Image height
     - Sampling steps
     - CFG scale
     - Sampler
     - Scheduler
     - Seed
     - control_after_generate (defines seed behavior)

    Outputs:
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

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
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

    RETURN_TYPES = ("LATENT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT",
                    "INT", "INT", "FLOAT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("LATENT", "SAMPLER", "SCHEDULER", "width", "height", "steps", "cfg", "sampler_name",
                    "scheduler", "seed")

    def process(self, image_width: int, image_height: int, sampling_steps: int,
                cfg: float, sampler_name: str, scheduler_name: str, seed: int) -> tuple:
        batch_size = 1
        latent = torch.zeros([batch_size, 4, image_height // 8, image_width // 8], device=self.device)
        LATENT = {"samples": latent}

        return (LATENT, sampler_name, scheduler_name, image_width, image_height, sampling_steps, cfg,
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

    def process(self, dimensions) -> tuple[int, int]:
        dim, orient = dimensions.split(' (', 1)
        # noinspection PyUnusedLocal
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


class TWCUI_Util_ModelVAELoader(BaseNode):
    def __init__(self):
        super().__init__()
        self.model_hashes: dict = {}
        self.vae_hashes: dict = {}

    def _load_hashes(self):
        try:
            with open(os.path.join(folder_paths.base_path, 'model_hashes.json'), 'r', encoding='utf-8') as f:
                print("TWCUI: model_hashes.json is present. Loading hashes from file.")
                self.model_hashes = json.load(f)
        except FileNotFoundError:
            print("TWCUI: model_hashes.json is not present. Not loading hashes, preparing new hash data.")
            # format: { "full path": "hashsum" }

        try:
            with open(os.path.join(folder_paths.base_path, 'vae_hashes.json'), 'r', encoding='utf-8') as f:
                print("TWCUI: vae_hashes.json is present. Loading hashes from file.")
                self.vae_hashes = json.load(f)
        except FileNotFoundError:
            print("TWCUI: vae_hashes.json is not present. Not loading hashes, preparing new hash data.")
            # format: { "full path": "hashsum" }

    @staticmethod
    def _calculate_sha256(file_path):
        """
        Calculates SHA256 sums of specified file paths.
        :param file_path: Path-like object, specifies a file path for opening.
        :return: 10-character string, the last 10 characters of the SHA256 hash.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to avoid loading the entire file into memory
            for byte_block in iter(lambda: f.read(4096), b""):
                # noinspection PyTypeChecker
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()[:10]

    @staticmethod
    def _load_checkpoint(ckpt_name, output_vae=False,
                         output_clip=True) -> tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP]:
        # Implementation taken from CheckpointLoaderSimple in ComfyUI nodes.py
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae, output_clip,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # `out` format with output_vae = False and output_clip = True:
        # (MODEL, CLIP, None, None)

        return out[0], out[1]

    @staticmethod
    def _load_taesd(name) -> dict:
        # Borrowed verbatim from comfyui's implementations
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
        return sd

    @staticmethod
    def _vae_list() -> list:
        # Borrowed verbatim from comfyui's implementations.
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        return vaes

    def _load_vae(self, vae_name: str) -> comfy.sd.VAE:
        # Load VAE
        if vae_name in ["taesd", "taesdxl"]:
            sd = self._load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        VAE = comfy.sd.VAE(sd=sd)

        return VAE

    def _get_checkpoint_hash(self, ckpt_name) -> str:
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if ckpt_path not in self.model_hashes.keys():
            print("TWCUI: Checkpoint not in known hash set, calculating checkpoint/model hash. "
                  "This may take a few moments.")
            self.model_hashes[ckpt_path] = self._calculate_sha256(ckpt_path)
        else:
            print("TWCUI: Checkpoint in known hashes.")

        return self.model_hashes[ckpt_path]

    def _get_vae_hash(self, vae_name) -> str:
        if vae_name is None or vae_name in ["taesd", "taesdxl"]:
            print("TWCUI: Current implementation cannot calculate hashes for TAESD or TAESDXL.")
            return "unknown"
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            if vae_path not in self.vae_hashes:
                print("TWCUI: VAE not in known hash set, calculating VAE hash. This may take a few moments.")
                self.vae_hashes[vae_path] = self._calculate_sha256(vae_path)
            else:
                print("TWCUI: VAE in known hashes.")

            return self.vae_hashes[vae_path]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (cls._vae_list(),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name", "model_hash", "vae_name", "vae_hash")

    CATEGORY = MODULE_CATEGORY

    def process(self, ckpt_name: str, vae_name: str) -> tuple:
        # Load hashes from files for checkpoint/models and VAEs.
        self._load_hashes()

        # Define variable types here.
        MODEL: comfy.model_patcher.ModelPatcher
        CLIP: comfy.sd.CLIP
        VAE: comfy.sd.VAE

        # load MODEL and CLIP
        MODEL, CLIP = self._load_checkpoint(ckpt_name)

        VAE = self._load_vae(vae_name)

        # Hashes!
        # First, check MODEL hash.
        model_hash = self._get_checkpoint_hash(ckpt_name)

        # Now, look at VAE.
        vae_hash = self._get_vae_hash(vae_name)

        return MODEL, CLIP, VAE, ckpt_name, model_hash, vae_name, vae_hash


class TWCUI_Util_ModelVAELORALoader(TWCUI_Util_ModelVAELoader):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (cls._vae_list(),),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "lora_str_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_str_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    def __init__(self):
        super().__init__()
        self.loaded_lora = None
        self.lora_hashes: dict = {}

    def _load_lora_hashes(self):
        try:
            with open(os.path.join(folder_paths.base_path, 'lora_hashes.json'), 'r', encoding='utf-8') as f:
                print("TWCUI: lora_hashes.json is present. Loading hashes from file.")
                self.model_hashes = json.load(f)
        except FileNotFoundError:
            print("TWCUI: lora_hashes.json is not present. Not loading hashes, preparing new hash data.")
            # format: { "full path": "hashsum" }

    def _get_lora_hash(self, lora_name) -> str:
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path not in self.lora_hashes.keys():
            print("TWCUI: Checkpoint not in known hash set, calculating checkpoint/model hash. "
                  "This may take a few moments.")
            self.lora_hashes[lora_path] = self._calculate_sha256(lora_path)
        else:
            print("TWCUI: Checkpoint in known hashes.")

        return self.lora_hashes[lora_path]

    def _load_lora(self, model, clip, lora_name, strength_model, strength_clip) -> tuple:
        if strength_model == 0 and strength_clip == 0:
            return model, clip

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return model_lora, clip_lora

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name", "model_hash", "vae_name", "vae_hash", "lora_name",
                    "lora_hash")

    # noinspection PyMethodOverriding
    def process(self, ckpt_name: str, vae_name: str, lora_name: str, lora_str_model: float,
                lora_str_clip: float) -> tuple:
        # Load hashes from files for checkpoint/models and VAEs.
        self._load_hashes()

        # Load LoRA hashes
        self._load_lora_hashes()

        # Define variable types here.
        cMODEL: comfy.model_patcher.ModelPatcher
        cCLIP: comfy.sd.CLIP
        lMODEL: comfy.model_patcher.ModelPatcher
        lCLIP: comfy.sd.CLIP
        VAE: comfy.sd.VAE

        # load MODEL and CLIP for Checkpoint
        cMODEL, cCLIP = self._load_checkpoint(ckpt_name)
        lMODEL, lCLIP = self._load_lora(cMODEL, cCLIP, lora_name, lora_str_model, lora_str_clip)

        # load VAE
        VAE = self._load_vae(vae_name)

        # Hashes!
        # First, check MODEL hash.
        model_hash = self._get_checkpoint_hash(ckpt_name)

        # Now, look at VAE.
        vae_hash = self._get_vae_hash(vae_name)

        # Now, LoRA hash!
        lora_hash = self._get_lora_hash(lora_name)

        return lMODEL, lCLIP, VAE, ckpt_name, model_hash, vae_name, vae_hash, lora_name, lora_hash
