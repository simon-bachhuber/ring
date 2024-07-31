import os
from pathlib import Path
from typing import Optional
import warnings


def parse_path(
    path: str,
    *join_paths: str,
    extension: Optional[str] = None,
    file_exists_ok: bool = True,
    mkdir: bool = True,
    require_is_file: bool = False,
) -> str:
    """Utility for performing actions on a path

    Args:
        path (str): The (root) path to file or folder
        extension (Optional[str], optional): The file extension. Defaults to None.
        file_exists_ok (bool, optional): If false, will error if file already exists.
            Defaults to True.
        mkdir (bool, optional): Will create all subdirs in the process.
            Defaults to True.
        require_is_file (bool, optional): if true, will error if file does not
            already exist. Defaults to False.

    Raises:
        Exception: _description_

    Returns:
        str: The home-expanded path to folder or file
    """
    path = Path(os.path.expanduser(path))

    for p in join_paths:
        path = path.joinpath(p)

    if extension is not None:
        if extension != "":
            extension = ("." + extension) if (extension[0] != ".") else extension

        # check for paths that contain a dot "." in their filename (through a number)
        # or that already have an extension
        old_suffix = path.suffix
        if old_suffix != "" and old_suffix != extension:
            warnings.warn(
                f"The path ({path}) already has an extension (`{old_suffix}`), but "
                f"it gets replaced by the extension=`{extension}`."
            )

        path = path.with_suffix(extension)

    if not file_exists_ok and os.path.exists(path):
        raise Exception(f"File {path} already exists but shouldn't")

    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)

    if require_is_file:
        assert path.is_file(), f"Not a file: {path}"

    return str(path)
