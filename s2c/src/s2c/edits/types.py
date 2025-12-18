
from __future__ import annotations
from typing import Literal, Optional, TypedDict, Union

JSONScalar = Union[str, int, float, bool, None]

class EditBase(TypedDict):
    op: str

class RenameEdit(EditBase):
    op: Literal["rename"]
    target_kind: Literal["var", "func", "class"]
    target: str
    new_name: str
    function: Optional[str]

class AddParamEdit(EditBase):
    op: Literal["add_param"]
    function: str
    name: str
    default: Optional[JSONScalar]

class WrapEdit(EditBase):
    op: Literal["wrap"]
    start_line: int
    end_line: int
    wrapper: Literal["try_except"]
