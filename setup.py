from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memvid",
    version="0.1.2",
    author="Memvid Team",
    author_email="team@memvid.ai",
    description="AI memory library using SQLite and FAISS for fast semantic search and retrieval", # Changed
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/olow304/memvid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # "Topic :: Multimedia :: Video", # Removed
        "Topic :: Database", # Added
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=2.2.0", # Keep
        "numpy>=1.21.0,<2.0.0",       # Keep
        "openai>=1.0.0",              # Keep
        "tqdm>=4.50.0",               # Keep (used in encoder/index)
        "faiss-cpu>=1.7.0",           # Keep
        "python-dotenv>=0.19.0",      # Keep (used by examples)
        # Removed: qrcode, opencv-python, opencv-contrib-python, Pillow
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "pdf": [
            "PyPDF2==3.0.1",
        ],
        "web": [
            "fastapi>=0.100.0",
            "gradio>=4.0.0",
        ],
    },
)