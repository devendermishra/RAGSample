from setuptools import setup, find_packages
from typing import List

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements() -> List[str]:
    """Read requirements from requirements.txt file.
    
    Returns:
        List of requirement strings
        
    Raises:
        FileNotFoundError: If requirements.txt file is not found
    """
    requirements: List[str] = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    except FileNotFoundError:
        raise FileNotFoundError("requirements.txt not found. Please ensure it exists in the project root.")
    return requirements

requirements = read_requirements()

setup(
    name="rag-sample",
    version="0.1.0",
    author="Devender Mishra",
    author_email="your.email@example.com",
    description="A sample command line (CLI) based conversational RAG app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devendermishra/RAGSample",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-sample=rag_sample.cli:main",
        ],
    },
)
