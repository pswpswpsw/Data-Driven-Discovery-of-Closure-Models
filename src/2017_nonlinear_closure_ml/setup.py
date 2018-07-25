from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [Extension("*", ["*.pyx"], extra_compile_args=['-O3'])]

setup(
    name='nonlinear_ML_closure',
    ext_modules = cythonize(extensions)
)
