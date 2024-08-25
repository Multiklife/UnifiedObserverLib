from setuptools import setup, find_packages

setup(
    name="unified-observer-lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "tensorflow",
        "tensorflow-quantum",
        "cirq",
        "matplotlib",
        "plotly",
        "ripser",
        "persim",
        "scikit-learn",
        "horovod",
        "cupy"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive library implementing the Unified Observer concept for AI and machine learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/unified-observer-lib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
