"""
Compatibility package configuration for BASICS-CDSS.

The canonical project metadata is defined in pyproject.toml; this file keeps
the repository structure consistent with the companion research repositories.
"""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


def read_readme() -> str:
    """Return the project README for package metadata."""
    return README.read_text(encoding="utf-8") if README.exists() else ""


setup(
    name="basics-cdss",
    version="2.1.0",
    author="Chatchai Tritham, Chakkrit Snae Namahoot",
    author_email="chatchait66@nu.ac.th, chakkrits@nu.ac.th",
    description=(
        "BASICS-CDSS: Simulation-based evaluation for safety-critical clinical "
        "decision support systems with XAI and clinical metrics"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ChatchaiTritham/BASICS-CDSS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "pydantic>=2",
        "shap>=0.42.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "rich"],
        "notebooks": ["jupyter", "jupyterlab", "ipywidgets"],
        "all": ["pytest", "pytest-cov", "rich", "jupyter", "jupyterlab", "ipywidgets"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
