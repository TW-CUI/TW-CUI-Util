"""Base functionality and nodes are incorporated here.

Primarily contains base class declarations inherited elsewhere.

Also has some global declarations."""


GLOBAL_CATEGORY = "TW-CUI"


class TWCUI_Util_BaseNode:
    """
    Base class for all custom ComfyUI nodes in this repository.

    Mostly done to makes sure that things're defined properly in
    any inherited functions during development.
    """
    def __init__(self, **kwargs) -> None:
        pass

    # noinspection PyPep8Naming
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        raise NotImplementedError

    RETURN_TYPES: tuple = ()
    RETURN_NAMES: tuple = ()

    CATEGORY: str = GLOBAL_CATEGORY

    FUNCTION: str = "process"

    # Defines if we're an Output Node or not
    OUTPUT_NODES: bool = False

    def process(self, **kwargs) -> tuple:
        raise NotImplementedError
