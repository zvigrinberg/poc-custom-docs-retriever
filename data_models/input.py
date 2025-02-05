
import typing
from uuid import uuid4

from pydantic import AliasChoices, BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import field_validator

from .common import HashableModel


class SourceDocumentsInfo(HashableModel):
    """
    Information about the source documents for the container image.

    - type: document type.
    - git_repo: git repo URL where the source documents can be cloned.
    - ref: git reference, such as tag/branch/commit_id
    - include: file extensions to include when indexing the source documents.
    - exclude: file extensions to exclude when indexing the source documents.
    """

    type: typing.Literal["code", "doc"]

    git_repo: typing.Annotated[str, Field(min_length=1)]
    ref: typing.Annotated[str, Field(min_length=1, validation_alias=AliasChoices(
        "ref", "tag"))]  # Support "tag" as alias for backward compatibility

    include: list[str] = ["*.py", "*.ipynb"]
    exclude: list[str] = []

    @field_validator("include", "exclude")
    @classmethod
    def sort_lists(cls, v: list[str]) -> list[str]:
        return list(sorted(v))



class ImageInfoInput(HashableModel):
    """
    Information about a container image, including the source information and sbom information.
    """
    name: str | None = None  # The image name
    tag: str | None = None  # i.e. latest
    digest: str | None = None  # i.e. sha256:...
    platform: str | None = None  # i.e. linux/amd64
    feed_group: str | None = None  # i.e. ubuntu:22.04

    source_info: list[SourceDocumentsInfo]




