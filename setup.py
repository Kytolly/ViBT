from setuptools import setup

setup(
    name="vibt",
    version="0.1.0",
    description="ViBT: Vision Bridge Transformer",
    packages=["vibt"],
    install_requires=[
        "opencv-python",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
        "einops",
        "peft"
    ],
)
