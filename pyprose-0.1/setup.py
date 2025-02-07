from setuptools import setup, Extension
from Cython.Build import cythonize
#import shutil
import numpy

compiler_directives = {
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "language_level": "3",
    "nonecheck": False,
    "initializedcheck": False,
    "cdivision_warnings": False,
    "embedsignature": True,
    "profile": False,
    "linetrace": False,
}

extensions = [
    Extension("pyprose.fd2d.elasticproperties", ["src/pyprose/fd2d/elasticproperties.pyx"], include_dirs=[numpy.get_include()]),
    Extension("pyprose.fd2d.acquisition", ["src/pyprose/fd2d/acquisition.pyx"], include_dirs=[numpy.get_include()]),
    Extension("pyprose.fd2d.wavefields", ["src/pyprose/fd2d/wavefields.pyx"], include_dirs=[numpy.get_include()]),
    Extension("pyprose.disp.dispersion", ["src/pyprose/disp/dispersion.pyx"], include_dirs=[numpy.get_include()]),
    Extension("pyprose.modgen.modelgenerator", ["src/pyprose/modgen/modelgenerator.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="pyprose",
    version="0.1",
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    packages=["pyprose", "pyprose.fd2d"],  # Inclure les packages dans la liste
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'scipy',
        'Cython',
        'mpi4py',
        'matplotlib'
    ],
)