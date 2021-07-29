from setuptools import setup, find_packages

xumx_version = "0.0.1"

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xumx_slicq",
    version=xumx_version,
    author="Sevag Hanssian",
    author_email="sevagh@pm.me",
    url="https://github.com/sevagh/xumx-sliCQ",
    description="sliCQ adaptation of the well-known Open-Unmix music source separation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.6",
    install_requires=["numpy", "torchaudio>=0.8.0", "torch>=1.8.0", "tqdm"],
    extras_require={
        "tests": [
            "pytest",
            "musdb>=0.4.0",
            "museval>=0.4.0",
            "onnx",
            "tqdm",
        ],
        "stempeg": ["stempeg"],
        "evaluation": ["musdb>=0.4.0", "museval>=0.4.0"],
    },
    entry_points={"console_scripts": ["umx=xumx_slicq.cli:separate"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
