from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
setup(
    name='emd_cuda',
    ext_modules=[
        CUDAExtension(
            name='emd_cuda',
            sources=[
                "/".join(__file__.split('/')[:-1] + ['emd.cpp']),
                "/".join(__file__.split('/')[:-1] + ['emd_kernel.cu']),
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
            # extra_compile_args={
            #     "cxx": ["-O3"],
            #     "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            # },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
