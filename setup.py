# setup.py
import setuptools
import re
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

PKG = "qics"
VERSIONFILE = os.path.join(PKG, "_version.py")
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="qics",
    version=verstr,
    author="Kerry He",
    author_email="he.kerry.k@gmail.com",
    description="Conic solver for quantum information theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kerry-he/qics",
    license="MIT",
    packages=setuptools.find_packages(include=["qics", "qics.*"]),
    python_requires=">=3",
    install_requires=["numpy", "scipy", "numba"],
    package_data={"": ["README.md", "LICENSE"]},
)
