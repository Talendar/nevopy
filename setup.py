# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" NEvoPY is an open source neuroevolution framework for Python.

NEvoPY is an open source software library that implements high performance
neuroevolution algorithms. It's flexible, easy to deploy and compatible with
large-scale distributed computing. It's been originally developed by Gabriel
Guedes Nogueira (Talendar).
"""

import setuptools

# Todo: FOLLOW SEMANTIC VERSIONING
_VERSION = "0.0.1"

# Short description.
short_description = "An open source neuroevolution framework for Python."

# Packages needed for nevopy to run.
# The compatible release operator (`~=`) is used to match any candidate version
# that is expected to be compatible with the specified version.
REQUIRED_PACKAGES = [
    "Columnar ~= 1.3.1",
    "matplotlib ~= 3.3.3",
    "mypy ~= 0.790",
    "networkx ~= 2.5",
    "numpy ~= 1.19.5",
    "pygraphviz ~= 1.6",  # todo: this requires graphviz to be installed
    "ray ~= 1.1.0",
]

# Packages which are only needed for testing code.
TEST_PACKAGES = [

]

# Loading the "long description" from the projects README file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nevopy",
    version=_VERSION,
    author="Gabriel Guedes Nogueira (Talendar)",
    author_email="no_email",  # todo: email
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Talendar/nevopy",
    download_url=None,  # todo: link to releases
    # Contained modules and scripts:
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    # PyPI package information:
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    license='MIT License',
    python_requires='>=3.6',
    keywords="nevopy neuroevolution evolutionary algorithms machine learning",
)
