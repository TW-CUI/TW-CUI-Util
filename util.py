from ._base import TWCUI_Util_BaseNode as BaseNode


class TWCUI_Util_StringLiteral(BaseNode):
    """
    Simple String value node that allows you to specify a string to pass
    into other nodes.

    Does not permit multiline text. See HelperNodes_MultilineStringLiteral
    for multiline text values.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "string": ("STRING", {"multiline": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)

    def process(self, string) -> tuple:
        return (string,)


class TWCUI_Util_MultilineStringLiteral(BaseNode):
    """
    Simple String value node that allows you to specify a string to pass
    into other nodes.

    This node permits multiline text.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "string": ("STRING", {"multiline": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)

    def process(self, string) -> tuple:
        return (string,)


class TWCUI_Util_IntLiteral(BaseNode):
    """
    Simple Integer value node that allows you to specify an integer to pass
    into other nodes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "value": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)

    def process(self, value) -> tuple:
        return (value,)


class TWCUI_Util_FloatLiteral(BaseNode):
    """
    Simple Float value node that allows you to specify an integer to pass
    into other nodes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0})
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)

    def process(self, value) -> tuple:
        return (value,)
