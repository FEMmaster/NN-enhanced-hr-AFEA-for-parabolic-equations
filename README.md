# NN-Enhanced hr-AFEA for Parabolic Equations

This repository contains the source code associated with the paper:

**J. Hao, Y. Huang, N. Yi, and P. Yin,  
"Neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations,"  
Journal of Computational Physics, 565 (2026), 115173.**

The repository contains both the original C++ implementation and a newly added Python implementation of the numerical algorithms.

## Repository Structure

```text
NN-enhanced-hr-AFEA-for-parabolic-equations/
├── Cpp_HR/
│   └── README.md
├── Python_HR/
│   └── README.md
└── README.md
```

- `Cpp_HR/` contains the original C++ implementation.
- `Python_HR/` contains the Python implementation.
- Detailed installation and execution instructions are provided in the corresponding README file of each subdirectory.

## Repository Update

The original version of this repository contained only the C++ implementation of the neural network-enhanced $hr$-adaptive finite element algorithm.

The repository has now been reorganized and a Python implementation has been added to improve the accessibility and reproducibility of the numerical experiments.

The main reasons for introducing the Python version are:

1. The original C++ implementation relies on a finite element library that is not publicly available, which makes it difficult for other users to reproduce the numerical experiments directly.

2. The C++ implementation uses LibTorch for the neural network components. Configuring LibTorch, particularly under Windows, can be relatively complicated and may require additional environment and library configuration.

3. The Python implementation provides a more accessible computational environment based on publicly available packages and is therefore more convenient for reproducing, testing, and extending the numerical experiments.

The reorganization of the repository is primarily intended to clearly separate the two implementations and provide implementation-specific documentation.

## Recommended Implementation

For reproducibility and ease of use, **we recommend using the Python implementation in `Python_HR/`**.

Detailed information about the required Python environment, dependencies, numerical examples, input parameters, and execution procedures can be found in:

```text
Python_HR/README.md
```

The Python version is intended to provide a more convenient and reproducible implementation of the numerical experiments without relying on the non-public finite element library used by the original C++ code.

## C++ Implementation

The original C++ implementation is retained in:

```text
Cpp_HR/
```

This version corresponds to the original implementation used in the development of the method.

Because it relies on a non-public finite element library and requires LibTorch configuration, it may not be straightforward to reproduce the numerical experiments in a new computational environment.

Additional information about the C++ source code and its configuration is provided in:

```text
Cpp_HR/README.md
```

## Python Implementation

The Python implementation is provided in:

```text
Python_HR/
```

This implementation was added to facilitate reproduction of the numerical experiments using a more accessible software environment.

For installation requirements, dependencies, directory structure, numerical examples, execution commands, and output descriptions, please refer to:

```text
Python_HR/README.md
```

Users who wish to reproduce the numerical results or further experiment with the method are encouraged to use this version.

## Citation

If you use the code in this repository, please cite:

```bibtex
@article{Hao2026,
  title   = {Neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations},
  volume  = {565},
  url     = {https://doi.org/10.1016/j.jcp.2026.115173},
  doi     = {10.1016/j.jcp.2026.115173},
  journal = {Journal of Computational Physics},
  publisher = {Elsevier BV},
  author  = {Hao, Jiaxiong and Huang, Yunqing and Yi, Nianyu and Yin, Peimeng},
  year    = {2026},
  month   = jul,
  pages   = {115173}
}
```
