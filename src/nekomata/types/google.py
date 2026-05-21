"""Google client specific types."""

from typing import Literal, NamedTuple

from nekomata.types.clients import PackageSpecificArgs


class InteractionsArgs(NamedTuple):
    """Arguments for the interactions API."""

    interaction_id: str | None = None
    store: bool = True


class GoogleArgs(PackageSpecificArgs):
    """Google specific API arguments."""

    api: Literal['generateContent', 'interactions'] = 'generateContent'

    interactions_args: InteractionsArgs | None = None
    """Arguments specific to the interactions API."""
