# Bypass line length limit
# ruff: noqa: E501

from typing import Annotated
from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, AliasGenerator, Field, field_validator


class ServeRequest(BaseModel):
    """Request that must be posted in order to get the prediction result from a served
    model (e.g. via FastAPI).
    """

    model_config = ConfigDict(
        alias_generator = AliasGenerator(
            # This will change "var_name" to "Var Name" when dumping the model
            alias = lambda var: var.replace('_', ' ').capitalize()
        ),
        # Allow request using the raw field name or the alias
        validate_by_name = True
    )

    # BUG: The uploaded file may not be converted as bytes automatically
    # Related link: https://github.com/fastapi/fastapi/discussions/12960
    image: UploadFile | bytes

    subject_focus: Annotated[bool, Field(description = 'My pet stands out against uncluttered background, not too close/far')] = False
    eyes: Annotated[bool, Field(description = 'My pet eyes are facing the front/near-front, with at least 1 clear eye/pupil')] = False
    face: Annotated[bool, Field(description = 'My pet face is captured pretty clearly, facing the front/near-front')] = False
    near: Annotated[bool, Field(description = 'My pet (alone) is taking a significant portion of the image dimension (>= 50%)')] = False
    action: Annotated[bool, Field(description = 'My pet is in the middle of an action (e.g. jumping)')] = False
    accessory: Annotated[bool, Field(description = 'My image has accompanying accessory (e.g. toy, sticker), excluding collar/leash')] = False
    group: Annotated[bool, Field(description = 'There are more than 1 pet in the image')] = False
    collage: Annotated[bool, Field(description = 'My image has been digitally-retouched (e.g. added frame, combined images)')] = False
    human: Annotated[bool, Field(description = 'There is a human in the image')] = False
    occlusion: Annotated[bool, Field(description = 'My image has undesirable object blocking the pet (e.g. human, cage)')] = False
    info: Annotated[bool, Field(description = 'My image has custom-added text or labels (e.g. pet name, description)')] = False
    blur: Annotated[bool, Field(description = 'My image is noticeably blurry or noisy, including the pet eyes/face')] = False

    @field_validator('image', mode = 'after')
    @classmethod
    def read_image_as_bytes(cls, value: UploadFile | bytes) -> bytes:
        # HACK: Temporary fix for the bug above
        if hasattr(value, 'file'):
            return value.file.read()
        return value

    def dump_form(self) -> dict[str, bool]:
        """Dump the form part of the request (excluding image) by alias."""

        # It seems that the model field order is always the same
        # So we can just pass this directly as a dataframe input later
        return self.model_dump(by_alias = True, exclude = 'image')


class ModelInfo(BaseModel):
    """Model info that generated the prediction result."""

    model_config = ConfigDict(validate_default = False)

    source: Annotated[str, Field(exclude = True)] = None
    version: str
    variant: str


class ServeResponse(BaseModel):
    """Response of the request, containing model info and prediction result."""

    model: ModelInfo
    result: float | list