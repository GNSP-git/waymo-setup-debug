from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="obb_distance",
        sources=["obb_distance.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="obb_distance",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=False,
    ),
)
