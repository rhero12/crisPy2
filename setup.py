import codecs, os, sys, re, setuptools
from sphinx.setup_command import BuildDoc

cmdclass = {"build_sphinx" : BuildDoc}

with open("README.md", "r") as fh:
    long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)

    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="crispy2",
    version=find_version("crispy/__init__.py"),
    author="John Armstrong",
    author_email="j.armstrong.2@research.gla.ac.uk",
    description="A Python package for using data from the Swedish 1 m Solar Telescope's CRisp Imaging SpectroPolarimeter instrument.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rhero12/crispy2",
    packages=setuptools.find_packages(),
    install_requires = [
        "torch",
        "astropy",
        "matplotlib",
        "ipywidgets",
        "jupyterlab",
        "h5py",
        "pyyaml",
        "tqdm",
        "cycler",
        "specutils",
        "numba"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">3.6",
    cmdclass=cmdclass,
    command_options={
        "build_sphinx" : {
            "project" : ("setup.py", "crispy"),
            "version" : ("setup.py", find_version("crispy/__init__.py")),
            "source_dir" : ("setup.py", "docs")
        }
    }
)
