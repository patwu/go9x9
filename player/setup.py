from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(Extension(
    "pyboard",
    sources=["pyboard.pyx"],
    language='c++',
    include_dirs=[r'.', r'../pachi', r'../boardwarp',np.get_include()],
    library_dirs=[r'../lib'],
    libraries=['pachi','board'],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)))

