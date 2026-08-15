````markdown
# NN-Enhanced hr-AFEA for Parabolic Equations

This repository contains the source code associated with the paper:

**J. Hao, Y. Huang, N. Yi, and P. Yin,  
"Neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations,"  
Journal of Computational Physics, 565 (2026), 115173.**

The repository contains the original C++ implementation and a newly added Python implementation **(added on 2026-08-15)** of the numerical algorithms.

For reproducibility and ease of use, **we recommend using the Python implementation**.

## Repository Structure

```text
NN-enhanced-hr-AFEA-for-parabolic-equations/
├── Cpp_HR/
│   └── README.md
├── Python_HR/
│   └── README.md
└── README.md
```

- `Cpp_HR/` contains the main source files from the original C++ implementation.
- `Python_HR/` contains the Python implementation developed to facilitate reproduction of the numerical experiments.
- Detailed information for each implementation is provided in the corresponding README file.

## Repository Update

The original version of this repository contained only the C++ implementation of the neural network-enhanced $hr$-adaptive finite element algorithm.

On **2026-08-15**, the repository was reorganized and a Python implementation was added to improve the accessibility and reproducibility of the numerical experiments.

The main reasons for introducing the Python version are:

1. The original C++ implementation relies on a finite element library that is not publicly available.

2. Because this finite element library cannot be distributed with the repository, only the main source files of the original C++ implementation are provided. Consequently, the C++ code is **not self-contained and cannot be run directly**, even if the other required software environment is successfully configured.

3. The C++ implementation also uses LibTorch for the neural network components. Configuring LibTorch, particularly under Windows, can be relatively complicated and requires additional library and environment configuration.

4. The Python implementation is based on publicly available software packages and provides a substantially more accessible environment for reproducing, testing, and extending the numerical experiments.

The repository was therefore reorganized to clearly separate the original C++ source files from the newly added Python implementation and to provide implementation-specific documentation for each version.

## Recommended Implementation

For reproducibility and ease of use, **users are strongly encouraged to use the Python implementation in `Python_HR/`**.

The Python version was introduced specifically to overcome the reproducibility limitations of the original C++ implementation. It uses publicly available software packages and does not depend on the non-public finite element library required by the C++ version.

Detailed information about the required Python environment, dependencies, directory structure, numerical examples, input parameters, execution procedures, and output files can be found in:

```text
Python_HR/README.md
```

Users who wish to reproduce the numerical results or further experiment with the method should use this version.

## C++ Implementation

The original C++ implementation is retained in:

```text
Cpp_HR/
```

This directory contains the main source files from the original implementation used in the development of the method.

However, the C++ implementation relies on a finite element library that is not publicly available and therefore cannot be included in this repository. As a result, the files provided in `Cpp_HR/` constitute only the main components of the original implementation and **do not form a directly executable standalone program**.

Thus, even after configuring the other required dependencies, including LibTorch, the C++ version cannot be run directly without access to the non-public finite element library and the associated supporting components.

The C++ source files are retained primarily to document the original implementation and provide a reference for the algorithmic structure.

Additional information about the C++ source files and their dependencies is provided in:

```text
Cpp_HR/README.md
```

## Python Implementation

The Python implementation, added on **2026-08-15**, is provided in:

```text
Python_HR/
```

This implementation was developed to provide a more accessible and reproducible version of the numerical experiments using publicly available software packages.

It is the **recommended implementation** for users who wish to reproduce the numerical results, examine the numerical algorithms, or conduct further experiments based on the method.

For installation requirements, dependencies, directory structure, numerical examples, execution commands, input parameters, and output descriptions, please refer to:

```text
Python_HR/README.md
```

## Citation

If you use the code in this repository, please cite:

```bibtex
@article{Hao2026,
  title     = {Neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations},
  volume    = {565},
  url       = {https://doi.org/10.1016/j.jcp.2026.115173},
  doi       = {10.1016/j.jcp.2026.115173},
  journal   = {Journal of Computational Physics},
  publisher = {Elsevier BV},
  author    = {Hao, Jiaxiong and Huang, Yunqing and Yi, Nianyu and Yin, Peimeng},
  year      = {2026},
  month     = jul,
  pages     = {115173}
}
```
````
