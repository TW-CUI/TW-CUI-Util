from .util import TWCUI_Util_StringLiteral, TWCUI_Util_MultilineStringLiteral
from .util import TWCUI_Util_IntLiteral, TWCUI_Util_FloatLiteral

NODE_CLASS_MAPPINGS = {
    "TWCUI_Util_FloatLiteral": TWCUI_Util_FloatLiteral,
    "TWCUI_Util_IntLiteral": TWCUI_Util_IntLiteral,
    "TWCUI_Util_StringLiteral": TWCUI_Util_StringLiteral,
    "TWCUI_Util_MultilineStringLiteral": TWCUI_Util_MultilineStringLiteral,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TWCUI_Util_FloatLiteral": "FLOAT Literal",
    "TWCUI_Util_IntLiteral": "INTEGER Literal",
    "TWCUI_Util_StringLiteral": "STRING Literal",
    "TWCUI_Util_MultilineStringLiteral": "STRING Literal (Multiline)",
}
