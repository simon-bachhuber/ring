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


subpkg_ml_requires = ["wandb", "neptune", "optax", "dm-haiku", "joblib"]
subpkg_omc_requires = ["qmt", "pandas"]
subpkg_datasets_requires = ["pyyaml", "joblib"]
render_mujoco_requires = ["mujoco"]
render_vispy_requires = ["vispy", "pyqt6"]
dev_requires = [
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings",
    "mkdocstrings-python",
    "mknotebooks",
    "pytest",
]


setuptools.setup(
    name="x_xy",
    packages=setuptools.find_packages(),
    version="0.8.2",
    package_data={
        "x_xy": find_data_files(
            "x_xy", patterns=["*.xml", "*.yaml", "*.joblib", "*.json"]
        ),
    },
    include_package_data=True,
    install_requires=[
        "jaxlib",
        "jax",
        "flax",
        "tqdm",
        "tree_utils @ git+https://github.com/SimiPixel/tree_utils.git",
    ],
    extras_require={
        "ml": subpkg_ml_requires,
        "omc": subpkg_omc_requires,
        "datasets": subpkg_datasets_requires,
        "render_mujoco": render_mujoco_requires,
        "render_vispy": render_vispy_requires,
        "dev": dev_requires,
        "all": subpkg_ml_requires
        + subpkg_omc_requires
        + subpkg_datasets_requires
        + render_mujoco_requires
        + render_vispy_requires,
    }
    # leave this comment in incase we need to knwo the syntax again in the future
    # entry_points={"console_scripts": ["xxy-render = x_xy.cli.render:main"]},
)
