# Python Implementation

This directory contains the Python implementation of the neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations.

The Python version was added on **2026-08-15** to improve the accessibility and reproducibility of the numerical experiments. It is the **recommended version** for reproducing and testing the numerical examples in this repository.

## Requirements

The code was developed and tested with:

- Python 3.8.20
- **FEALPy 1.1.20**
- PyTorch 2.4.1

Other required Python packages and their tested versions are listed in:

```text
requirements.txt
```

## Important Note on FEALPy

**FEALPy 1.1.20 is required for the current implementation.**

The code was developed using the API of this legacy version of FEALPy:

```text
fealpy==1.1.20
```

Users are strongly recommended to use this specific version. Newer versions of FEALPy may use different finite element interfaces and therefore may not run the code without modification.

## C++ Implementation

The original C++ implementation is provided in:

```text
../Cpp_HR/
```

Because the C++ version relies on a non-public finite element library, users who wish to reproduce the numerical experiments are recommended to use the Python version provided here.

See `../Cpp_HR/README.md` for further information.

## Reference

J. Hao, Y. Huang, N. Yi, and P. Yin,  
**"Neural network-enhanced $hr$-adaptive finite element algorithm for parabolic equations,"**  
*Journal of Computational Physics*, 565 (2026), 115173.  
https://doi.org/10.1016/j.jcp.2026.115173

If you use this code in academic work, please cite:

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
