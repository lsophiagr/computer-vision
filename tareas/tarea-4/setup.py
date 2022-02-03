from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import cv2

#numpy.get_include()

setup(ext_modules = cythonize('sharp.pyx'),
        include_dirs=[numpy.get_include()])