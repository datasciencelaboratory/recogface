from setuptools import setup, find_packages

setup(
    name="recogface",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["main"],
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe",
        "qdrant-client",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "recogface=recogface.cli:main",
        ],
    },
)
