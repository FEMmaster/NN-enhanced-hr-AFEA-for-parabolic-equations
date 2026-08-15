# C++ Implementation

This directory contains the main source files of the original C++ implementation of the neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations.

## Important Note

The original C++ implementation relies on a finite element library that is not publicly available and therefore cannot be distributed with this repository. As a result, only the main source files related to the proposed algorithm are provided here.

The missing part mainly concerns the underlying finite element computations. After configuring the required external dependencies, users only need to construct their own finite element computation module, or replace the original non-public finite element library with a finite element library of their choice, to use the provided main source files.

Therefore, the files in this directory are not directly executable as distributed, but they contain the main implementation of the algorithm and can be used once the required finite element functionality is supplied.

## External Dependencies

The original C++ implementation uses the following external software:

- **Gmsh** for mesh-related functionality  
  https://gmsh.info/

- **LibTorch** for the neural network components  
  https://pytorch.org/

In addition, the original implementation uses a non-public finite element library for the finite element computations. This library is not included in the repository.

To use the C++ version, users should:

1. Configure Gmsh and LibTorch in their local environment.
2. Provide the finite element computation components required by the main source files, either by implementing them directly or by adapting another finite element library.
3. Connect these components to the provided source files following the corresponding interfaces and data structures.

Once the finite element computation part is supplied, the main algorithmic components provided in this directory can be used normally.

## Recommended Version

For users who wish to reproduce the numerical experiments directly, we recommend using the Python implementation provided in:

```text
../Python_HR/
```

The Python version was added on **2026-08-15** to improve the accessibility and reproducibility of the numerical experiments. It is based on publicly available software packages and provides a complete implementation of the finite element and neural network computations required by the numerical experiments.

Detailed installation and execution instructions are provided in:

```text
../Python_HR/README.md
```

## Reference

The code in this directory is associated with:

J. Hao, Y. Huang, N. Yi, and P. Yin,  
**"Neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations,"**  
*Journal of Computational Physics*, 565 (2026), 115173.  
https://doi.org/10.1016/j.jcp.2026.115173
