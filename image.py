import os
import json

from datetime import datetime, timezone

from ._base import TWCUI_Util_BaseNode as BaseNode, GLOBAL_CATEGORY

# noinspection PyUnresolvedReferences
import folder_paths
# noinspection PyPackageRequirements, PyUnresolvedReferences
from nodes import MAX_RESOLUTION

# noinspection PyPackageRequirements, PyUnresolvedReferences
import comfy.samplers

# noinspection PyUnresolvedReferences
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import piexif
import piexif.helper
import numpy as np


MODULE_CATEGORY = f"{GLOBAL_CATEGORY}/image"


class TWCUI_Util_SaveImage(BaseNode):
    """
    An Output node that does a basic save of an image with optional basic ComfyUI metadata.
    """

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    RETURN_TYPES = ()

    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = MODULE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_%time", "multiline": False}),
            },
            "optional": {
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S-UTC", "multiline": False}),
                "save_metadata": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @staticmethod
    def _get_timestamp(time_format):
        now = datetime.now(tz=timezone.utc)
        # noinspection PyBroadException
        try:
            timestamp = now.strftime(time_format)
        except Exception:
            timestamp = now.strftime("%Y-%m-%d-%H%M%SUTC")

        return timestamp

    def save_images(self, images, filename_prefix: str, time_format: str, save_metadata: bool,
                    prompt: dict = None, extra_pnginfo: dict = None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if save_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            filename_with_time = filename_with_batch_num.replace("%time", self._get_timestamp(time_format))
            file = f"{filename_with_time}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {
            "ui": {
                "images": results
            }
        }


class TWCUI_Util_SaveImageAdvanced(BaseNode):
    """
    An Output node that does more advanced saving of images, and allows for optional inclusion of metadata.
    """

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_output_directory()

    RETURN_TYPES = ()

    FUNCTION = "save"

    OUTPUT_NODE = True

    CATEGORY = MODULE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        inputs = {
            "required": {
                "images": ("IMAGE", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "extension": (['png', 'jpeg', 'webp'], {"default": "png"}),
                "steps": ("INT", {"forceInput": True}),
                "cfg": ("FLOAT", {"forceInput": True}),
                "model_name": ("STRING", {"forceInput": True}),
                "vae_name": ("STRING", {"forceInput": True}),
                "sampler_name": ("STRING", {"forceInput": True}),
                "scheduler": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "positive_prompt": ("STRING", {"default": "unknown", "multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"default": "unknown", "multiline": True, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                                 "forceInput": True}),
                "width": ("INT", {"default": 1024, "min": 8, "max": MAX_RESOLUTION, "step": 8, "forceInput": True}),
                "height": ("INT", {"default": 1024, "min": 8, "max": MAX_RESOLUTION, "step": 8, "forceInput": True}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "save_workflow_with_metadata": ("BOOLEAN", {"default": False}),
                "save_extra_pnginfo_with_metadata": ("BOOLEAN", {"default": False}),
                "model_hash": ("STRING", {"default": "unknown", "forceInput": True}),
                "vae_hash": ("STRING", {"default": "unknown", "forceInput": True}),
                "compression": ("INT", {"default": 5, "min": 1, "max": 9, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

        return inputs

    @staticmethod
    def _get_timestamp(time_format):
        now = datetime.now(tz=timezone.utc)
        # noinspection PyBroadException
        try:
            timestamp = now.strftime(time_format)
        except Exception:
            timestamp = now.strftime("%Y-%m-%d-%H%M%SUTC")

        return timestamp

    def _make_pathname(self, filename: str, seed: int, modelname: str, time_format: str, batch_number: int,
                       counter: int = None) -> str:
        filename = filename.replace("%date", self._get_timestamp("%Y-%m-%d"))
        filename = filename.replace("%time", self._get_timestamp(time_format))
        filename = filename.replace("%model", modelname)
        filename = filename.replace("%seed", str(seed))
        filename = filename.replace("%batch_num%", str(batch_number))
        if counter:
            filename = filename.replace("%counter", f"{counter:05}")
        else:
            filename = filename.replace("%counter", "")
        return filename

    def _make_filename(self, filename: str, seed: int, modelname: str, time_format: str, batch_number: int,
                       counter: int = None) -> str:
        filename = self._make_pathname(filename, seed, modelname, time_format, batch_number, counter)

        return self._get_timestamp(time_format) if filename == "" else filename

    def save(self, images, filename_prefix: str, path: str, extension: str, steps: int, cfg: float,
             model_name: str, vae_name: str, sampler_name: str, scheduler: str, positive_prompt: str,
             negative_prompt: str, seed: int, width: int, height: int, lossless_webp: bool,
             quality_jpeg_or_webp: str, time_format: str, save_metadata: bool, save_workflow_with_metadata: bool,
             save_extra_pnginfo_with_metadata: bool, model_hash: str, vae_hash: str, compression: int,
             prompt: dict = None, extra_pnginfo: dict = None):
        if path or path == '':
            path = os.path.join(self.output_dir, path)
        else:
            path = self.output_dir

        if path.strip() != '':
            if not os.path.exists(path.strip()):
                print(f"The specified path `{path.strip()}` does not exist. Creating directory.")
                os.makedirs(path, exist_ok=True)

        subfolder = os.path.normpath(path)

        paths = []

        for (batch_number, image) in enumerate(images):
            counter = 1
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            parameters = (f"Prompt: {positive_prompt}. \nNegative prompt: {negative_prompt}. \nSteps: {steps}, "
                          f"Sampler: {sampler_name}, Scheduler: {scheduler}, CFG scale: {cfg}, Seed: {seed}, "
                          f"Size: {width}x{height}, Model hash: {model_hash}, Model: {model_name}, "
                          f"VAE hash: {vae_hash}, VAE: {vae_name}, Version: ComfyUI")

            filename_prefix = self._make_filename(filename_prefix, seed, model_name, time_format, batch_number,
                                                  counter)

            if extension == "png":
                metadata = PngInfo()
                if save_metadata:
                    metadata.add_text("parameters", parameters)
                    if prompt is not None and save_workflow_with_metadata:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None and save_extra_pnginfo_with_metadata:
                        for x in extra_pnginfo:
                            if x.lower() == 'workflow' and not save_workflow_with_metadata:
                                continue
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename = f"{filename_prefix}.png"
                file = os.path.join(self.output_dir, path, filename)
                img.save(file, pnginfo=metadata, compress_level=compression)
            else:
                filename = f"{filename_prefix}.{extension}"
                file = os.path.join(self.output_dir, path, filename)
                img.save(file, compress_level=compression, quality_jpeg_or_webp=quality_jpeg_or_webp,
                         lossless=lossless_webp,)
                if save_metadata:
                    exif_bytes = piexif.dump({
                        "Exif": {
                            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters, encoding="unicode")
                        },
                    })
                    piexif.insert(exif_bytes, file)

            paths.append(filename)
            counter += 1

        return {"ui": {"images": map(
            lambda fname: {"filename": fname, "subfolder": subfolder if subfolder != '.' else '',
                           "type": 'output'}, paths)}}
