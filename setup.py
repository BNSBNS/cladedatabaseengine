"""Setup configuration for Clade Database Engine."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cladedatabaseengine",
    version="0.1.0",
    author="Clade DB Team",
    description="A production-grade DBMS supporting OLTP and OLAP workloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cladedatabaseengine",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Database :: Database Engines/Servers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyarrow>=12.0",
        "numba>=0.57",
        "structlog",
        "typing-extensions",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "mypy",
            "pylint",
            "memory-profiler",
        ],
    },
)
