import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nevopy",
    version="0.0.1",
    author="Gabriel Nogueira (Talendar)",
    author_email="no_email",
    description="A neuroevolution framework for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Talendar/nevopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.6',
)