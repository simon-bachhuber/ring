import os
import subprocess


def setup_colab_env() -> bool:
    """Copied and modified from the getting-started-notebook of mujoco.
    Returns true if there is a colab context, else false.
    """
    try:
        from google.colab import files  # noqa: F401
    except ImportError:
        return False

    if subprocess.run("nvidia-smi", shell=True).returncode:
        raise RuntimeError(
            "Cannot communicate with GPU. "
            "Make sure you are using a GPU Colab runtime. "
            "Go to the Runtime menu and select Choose runtime type."
        )

    # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
    # This is usually installed as part of an Nvidia driver package, but the Colab
    # kernel doesn't install its driver via APT, and as a result the ICD is missing.
    # (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write(
                """{
            "file_format_version" : "1.0.0",
            "ICD" : {
                "library_path" : "libEGL_nvidia.so.0"
            }
        }
        """
            )

    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ["MUJOCO_GL"] = "egl"

    # install mediapy
    os.system("command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)")
    os.system("pip install -q mediapy")

    # install mujoco
    os.system("pip install -q mujoco")

    return True
