"""Setup script for the V-BIG package."""

from setuptools import setup, find_packages

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Variance-Based Integrated Gradients for Robust Natural Language Inference"

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vbig",
    version="0.1.0",
    author="Julian Weaver",
    author_email="julian.weaver@utexas.edu",
    description="Variance-Based Integrated Gradients for Robust Natural Language Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weavejul/V-BIG",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "pre-commit>=2.15",
        ],
        "wandb": ["wandb>=0.12"],
        "viz": ["seaborn>=0.11"],
    },
    entry_points={
        "console_scripts": [
            "vbig-train=examples.train_vbig_model:main",
            "vbig-analyze=examples.analyze_attributions:main",
            "vbig-compare=examples.compare_models:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)