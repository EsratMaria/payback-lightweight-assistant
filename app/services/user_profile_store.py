"""User profile store — resolves user_id to UserContext.

For the demo: backed by data/user_profiles.json.

"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.models.schemas import UserContext

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "user_profiles.json"


class UserProfileStore:
    """Loads user profiles from a JSON file. Singleton via get_default_store()."""

    def __init__(self, source_path: str | Path = _DEFAULT_PATH) -> None:
        path = Path(source_path)
        raw: dict = json.loads(path.read_text(encoding="utf-8"))
        self._profiles: dict[str, UserContext] = {}
        for uid, data in raw.items():
            try:
                self._profiles[uid] = UserContext.model_validate(data)
            except ValidationError as exc:
                logger.warning("Skipping malformed profile %r: %s", uid, exc)

    def get(self, user_id: str) -> Optional[UserContext]:
        """Return UserContext for the user_id, or None if not found."""
        return self._profiles.get(user_id)

    def list_user_ids(self) -> list[str]:
        """Return all known user_ids."""
        return list(self._profiles.keys())

    def list_profiles(self) -> dict[str, UserContext]:
        """Return all profiles keyed by user_id."""
        return dict(self._profiles)


@lru_cache(maxsize=1)
def get_default_store() -> UserProfileStore:
    return UserProfileStore()
