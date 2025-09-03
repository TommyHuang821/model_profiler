from setuptools import setup, find_packages

setup(
    name="model_profiler",
    version="0.1.0",
    author="Chih-Sheng (Tommy) Huang",
    author_email="chih.sheng.huang821@gmail.com",  # <-- 建議換成你的 email
    description="A lightweight PyTorch profiler for FLOPs, memory, and parameters, with Excel and Torchview visualization.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TommyHuang821/model_profiler",  # <-- 換成你的 GitHub repo
    packages=find_packages(exclude=("examples", "tests")),
    install_requires=[
        "torch>=1.10",
        "torchvision",
        "prettytable",
        "openpyxl",
        "torchview",
        "graphviz"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
)
