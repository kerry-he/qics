# setup.py
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qics",
    version="0.0",
    author="Kerry He",
    author_email="he.kerry.k@gmail.com",
    description="Conic solver for quantum information theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kerry-he/qics",
    license="MIT",
    
    packages=setuptools.find_packages(include=['qics','qics.*']),
    python_requires='>=3',
    install_requires=["numpy","scipy", "numba"],
    package_data={"": ["README.md","LICENSE"]}
)