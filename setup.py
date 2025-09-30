from setuptools import setup, find_packages

setup(
    name="spike-adaptive-llm",
    version="0.1.0",
    description="Context-Adaptive Spike Encoding for Large Language Models",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
    ],
    python_requires=">=3.8",
)