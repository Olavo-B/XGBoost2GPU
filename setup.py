#!/usr/bin/env python3
"""
Setup script for XGBoost2GPU library.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
def get_version():
    version_file = os.path.join(this_directory, 'src', 'xgboost2gpu', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read requirements
def get_requirements():
    requirements_file = os.path.join(this_directory, 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Handle git dependencies
                    if '@' in line and 'git+' in line:
                        # Extract package name for git dependencies
                        pkg_name = line.split('@')[0].strip()
                        requirements.append(pkg_name)
                    else:
                        requirements.append(line)
    return requirements

setup(
    name="xgboost2gpu",
    version=get_version(),
    author="Olavo Alves Barros Silva",
    author_email="olavo.barros@ufv.com",
    description="A library to generate CUDA code from XGBoost models using TreeLUT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Olavo-B/XGBoost2GPU",
    project_urls={
        "Bug Tracker": "https://github.com/Olavo-B/XGBoost2GPU/issues",
        "Documentation": "https://github.com/Olavo-B/XGBoost2GPU/blob/main/README.md",
        "Source Code": "https://github.com/Olavo-B/XGBoost2GPU",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "xgboost>=1.4.0",
        "scikit-learn>=1.0.0",
    ] + get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
        ],
        "examples": [
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
        ],
        "all": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0", 
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "xgboost2gpu=xgboost2gpu.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "xgboost2gpu": ["templates/*.cu", "templates/*.h"],
    },
    zip_safe=False,
    keywords=[
        "xgboost", 
        "cuda", 
        "gpu", 
        "machine learning", 
        "inference", 
        "acceleration",
        "treelut",
        "quantization"
    ],
)
