import os
from pathlib import Path
from typing import Optional


def parse_path(
    path: str,
    extension: Optional[str] = None,
    file_exists_ok: bool = True,
    mkdir: bool = True,
) -> str:
    path = Path(os.path.expanduser(path))

    if extension:
        extension = "." + extension if extension[0] != "." else extension
        path = path.with_suffix(extension)

    if not file_exists_ok and os.path.exists(path):
        raise Exception(f"File {path} already exists but shouldn't")

    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)
