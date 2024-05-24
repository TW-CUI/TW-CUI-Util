import os
import json

from datetime import datetime, timezone

from ._base import TWCUI_Util_BaseNode as BaseNode, GLOBAL_CATEGORY

import folder_paths
# noinspection PyPackageRequirements
from nodes import MAX_RESOLUTION

# noinspection PyPackageRequirements
import comfy.samplers

from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import piexif
import piexif.helper
import numpy as np


MODULE_CATEGORY = f"{GLOBAL_CATEGORY}/image"


class TWCUI_Util_SaveImage(BaseNode):
    """
    An Output node that does a quick save of an image without metadata.
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
        try:
            timestamp = now.strftime(time_format)
        except:
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
            filename_with_time = filename.replace("%time", self._get_timestamp(time_format))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
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