import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="Maestro",  # Replace with your own username
    version="0.0.1",
    author="Junlin Wang",
    author_email="junliw1@uci.edu",
    description="A package for adversarial attacks and defenses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ucinlp/maestro",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
