import os
from setuptools import setup, Extension
import numpy

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


levmar_sources = [
        os.path.join('cam_server/pipeline/data_processing/levmar-2.6', c)
        for c in ['lm.c', 'Axb.c', 'misc.c', 'lmlec.c', 'lmbc.c', 'lmblec.c', 'lmbleic.c']
        ]
levmar_sources += [
        'cam_server/pipeline/data_processing/levmar_c.c'
]

setup(name="cam_server",
      version="2.4.2",
      maintainer="Paul Scherrer Institute",
      maintainer_email="daq@psi.ch",
      author="Paul Scherrer Institute",
      author_email="daq@psi.ch",
      description="Camera server to convert epics enabled cameras into bsread cameras.",

      license="GPL3",

      ext_modules =[
          Extension('cam_server.pipeline.data_processing.levmar_c',
                sources=levmar_sources,
		include_dirs = [numpy.get_include()],
                extra_compile_args=['-std=c99', '-Wno-unused-but-set-variable', '-funroll-loops', '-fopenmp', '-march=native', '-O3'],
                libraries=['gomp', 'mkl_rt']
                )
          ],
      packages=['cam_server',
                "cam_server.camera",
                "cam_server.camera.rest_api",
                "cam_server.camera.source",
                "cam_server.instance_management",
                "cam_server.pipeline",
                "cam_server.pipeline.data_processing",
                "cam_server.pipeline.rest_api"],

      # long_description=read('Readme.md'),
      )
