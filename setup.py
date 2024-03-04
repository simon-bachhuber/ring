import fnmatch
import os

import setuptools


def find_data_files(package_dir, patterns, excludes=()):
    """Recursively finds files whose names match the given shell patterns."""
    paths = set()

    def is_excluded(s):
        for exclude in excludes:
            if fnmatch.fnmatch(s, exclude):
                return True
        return False

    for directory, _, filenames in os.walk(package_dir):
        if is_excluded(directory):
            continue
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # NB: paths must be relative to the package directory.
                relative_dirpath = os.path.relpath(directory, package_dir)
                full_path = os.path.join(relative_dirpath, filename)
                if not is_excluded(full_path):
                    paths.add(full_path)
    return list(paths)


dev_requires = [
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings",
    "mkdocstrings-python",
    "mknotebooks",
    "pytest",
    # for parallel test execution; $ pytest -n auto
    "pytest-xdist",
    # for testing of notebooks; $ pytest --nbmake **/*ipynb
    "nbmake",
]


setuptools.setup(
    name="x_xy",
    packages=setuptools.find_packages(),
    version="0.11.25",
    package_data={
        "x_xy": find_data_files(
            # parameters and datasets are now downloaded on-demand
            # but could exclude with exludes = ["**/exp/*", "**/pretrained/*"]
            package_dir="x_xy",
            patterns=["*.xml", "*.yaml", "*.json"],
            excludes=[],
        ),
    },
    include_package_data=True,
    install_requires=[
        "jaxlib",
        "jax",
        "jaxopt",
        "numpy",
        "flax",
        "tqdm",
        "wget",
        "h5py",
        "tree_utils @ git+https://github.com/SimiPixel/tree_utils.git",
        "wandb",
        "optax",
        "dm-haiku",
        "joblib",
        "pyyaml",
        "qmt",
        "scipy",
    ],
    extras_require={
        "dev": dev_requires,
    },
    # leave this comment in incase we need to knwo the syntax again in the future
    # entry_points={"console_scripts": ["xxy-render = x_xy.cli.render:main"]},
)
