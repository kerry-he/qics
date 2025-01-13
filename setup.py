# setup.py
import os
import re
import sys

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

PKG = "qics"
VERSIONFILE = os.path.join(PKG, "_version.py")
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

print(sys.version_info)
print(sys.version_info[:2])

if sys.version_info[:2] == (3, 13):
    NUMBA_VERSION = "numba==0.61.0rc2"
else:
    NUMBA_VERSION = "numba"

setuptools.setup(
    name="qics",
    version=VERSION,
    author="Kerry He, James Saunderson, and Hamza Fawzi",
    author_email="he.kerry.k@gmail.com, james.saunderson@monash.edu, h.fawzi@damtp.cam.ac.uk",
    description="Conic solver for quantum information theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    project_urls={
        "Source": "https://github.com/kerry-he/qics",
        "Documentation": "https://qics.readthedocs.io/en/latest/",
    },
    packages=setuptools.find_packages(include=["qics", "qics.*"]),
    python_requires=">=3.8",
    install_requires=["numpy", "scipy", "numba==0.61.0rc2"],
    package_data={"": ["README.md", "LICENSE.md"]},
)
