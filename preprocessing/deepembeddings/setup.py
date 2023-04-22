import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepembeddings",  # Replace with your own username
    version="0.0.3",
    author="Walber de Macedo Rodrigues",
    author_email="wmr@cin.ufpe.br",
    description="Deep embedding networks, tripletloss and siamese network.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
