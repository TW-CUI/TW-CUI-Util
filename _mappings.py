from .generation import TWCUI_Util_GenerationParameters, TWCUI_Util_CommonSDXLResolutions
from .generation import TWCUI_Util_GenerationPrompts

from .image import TWCUI_Util_SaveImage, TWCUI_Util_SaveImageAdvanced

from .util import TWCUI_Util_StringLiteral, TWCUI_Util_MultilineStringLiteral
from .util import TWCUI_Util_IntLiteral, TWCUI_Util_FloatLiteral, TWCUI_Util_BoolLiteral

NODE_CLASS_MAPPINGS = {
    # Generation category
    "TWCUI_Util_CommonSDXLResolutions": TWCUI_Util_CommonSDXLResolutions,
    "TWCUI_Util_GenerationParameters": TWCUI_Util_GenerationParameters,
    "TWCUI_Util_GenerationPrompts": TWCUI_Util_GenerationPrompts,

    # Image category
    "TWCUI_Util_SaveImage": TWCUI_Util_SaveImage,
    "TWCUI_Util_SaveImageAdvanced": TWCUI_Util_SaveImageAdvanced,

    # Util category
    "TWCUI_Util_BoolLiteral": TWCUI_Util_BoolLiteral,
    "TWCUI_Util_FloatLiteral": TWCUI_Util_FloatLiteral,
    "TWCUI_Util_IntLiteral": TWCUI_Util_IntLiteral,
    "TWCUI_Util_StringLiteral": TWCUI_Util_StringLiteral,
    "TWCUI_Util_MultilineStringLiteral": TWCUI_Util_MultilineStringLiteral,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Generation category
    "TWCUI_Util_CommonSDXLResolutions": "Common SDXL Resolutions",
    "TWCUI_Util_GenerationParameters": "Generation Parameters",
    "TWCUI_Util_GenerationPrompts": "Prompts",

    # Image category
    "TWCUI_Util_SaveImage": "Save Image",
    "TWCUI_Util_SaveImageAdvanced": "Save Image (Advanced)",

    # Util category
    "TWCUI_Util_BoolLiteral": "BOOLEAN Literal",
    "TWCUI_Util_FloatLiteral": "FLOAT Literal",
    "TWCUI_Util_IntLiteral": "INTEGER Literal",
    "TWCUI_Util_StringLiteral": "STRING Literal",
    "TWCUI_Util_MultilineStringLiteral": "STRING Literal (Multiline)",
}
