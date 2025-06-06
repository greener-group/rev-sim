# Reversible molecular simulation

This repository contains code and data from the paper:

- Greener JG. Reversible molecular simulation for training classical and machine learning force fields, [Proceedings of the National Academy of Sciences](https://www.pnas.org/doi/10.1073/pnas.2426058122) 122(22), e2426058122 (2025)

Please cite the paper if you use the method or code.
The licence is MIT.
The code makes use of [Molly.jl](https://github.com/JuliaMolSim/Molly.jl).

## Installation

The code was run with Julia v1.10.3 and the following package versions:
- Molly v0.21.1.
- Enzyme v0.12.24.
- StaticArrays v1.9.7.

Other versions may or may not work.
The [MACE-OFF23](https://github.com/ACEsuit/mace-off) model was run on a GPU with the following:
- PythonCall v0.9.20.
- MACE and PyTorch should be installed in a conda environment and the environment activated. Example commands:
```bash
conda create --name maceoff python==3.11
conda activate maceoff
pip install torch # Installed v2.6.0
pip install mace-torch
```

The diamond model was run on a GPU with the following:
- PythonCall v0.9.20.
- GPUCompiler v0.26.5 with [this change](https://github.com/JuliaGPU/GPUCompiler.jl/pull/556/commits/0e00885f9c3d54a6b999e84d58d6ac6cfbdc0023) in order for higher order AD to work.
- [difftre](https://github.com/tummfm/difftre), a Python package that uses Jax, should be installed in a conda environment and the environment activated. Example commands:
```bash
conda create --name difftre python=3.9
conda activate difftre
pip install chex==0.0.9 MDAnalysis jax[cuda111]==0.2.17 jax-md==0.1.13 optax==0.0.9 dm-haiku==0.0.4 sympy==1.8 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install ipython jupyterlab scipy==1.7
conda install cudatoolkit-dev=11.1 -c conda-forge
pip install -e . # From difftre directory (https://github.com/tummfm/difftre)
```

## Running

By default each script trains the relevant model with reversible simulation.
For example, to train the water model on 32 threads from the `water` directory:
```bash
julia -t 32 water.jl
```
The output directory is an optional argument.
Code for DMS (reverse mode AD) and training with ensemble reweighting is included, plus code for running validation simulations and benchmarks with references to the relevant figures.
Trained models and force field files are also included.

The diamond model occasionally runs into a PyGIL error due to a known issue with PythonCall.
[This may be fixed on later PythonCall versions](https://juliapy.github.io/PythonCall.jl/stable/faq/#Is-PythonCall/JuliaCall-thread-safe?).
