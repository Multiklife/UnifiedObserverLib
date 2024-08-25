from setuptools import setup, find_packages

setup(
    name="unified_observer_lib",
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
        "horovod"
    ],
    author="Multiklife",
    author_email="multiklife@icloud.com",
    description="A comprehensive library implementing the Unified Observer concept for AI and machine learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/multiklife/unified_observer_lib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
