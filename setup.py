import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simitate_dataset",
    version="0.0.1",
    author="Raphael Memmesheimer",
    author_email="raphael@uni-koblenz.de",
    description="Simitate dataset package for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/airglow/simitate_dataset_pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)
