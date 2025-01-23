# Nonperturbative Nonlinear Transport in a Floquet-Weyl Semimetal

This repository contains Python codes designed to study Weyl nodes in Td-MoTe2 driven by a laser field. By utilizing the Floquet formalism and Wannier90 data, the code allows the user to compute bulk bands and track the Weyl nodes in the driven state through k-space. This work was conducted as theoretical support for https://arxiv.org/pdf/2409.04531.


## Folder Structure

- **`src/`**: This folder contains the source code, including the Python scripts for computing bulk bands and tracking Weyl nodes.
- **`data/`**: This folder stores the Wannier90 output files required for the computations.
- **`results/`**: The folder where the outputs of the computations will be saved, including bulk band plots and Weyl node tracking data.

## Project Overview

The project consists of three Python scripts:

1. **`bulk_bands.py`**  
   Generates the bulk bands of the material using Wannier90 data.

2. **`search_node.py`**  
   Follows the Weyl nodes as they move in k-space under the influence of periodic driving.  

3. **`floquet_functions.py`**  
   Contains utility functions used by `bulk_bands.py` and `search_node.py`. This script does not contain any executable code on its own.


## Requirements

The following Python packages are required to run the scripts:

- `numpy`
- `matplotlib`
- `numba`
- `mpi4py`
- `pyyaml`
- `pickle`

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## File Descriptions

### `bulk_bands.py`
This script computes and plots the bulk band structure of the material based on Wannier90 data. Both codes are parallelized, allowing them to be run using MPI.

#### Usage
Simply run the script in your terminal to generate bulk bands. Output plots will be saved automatically.

```bash
mpirun -n <number_of_cores> python3 bulk_bands.py
```

### `search_node.py`
Tracks the Weyl nodes in k-space using the Floquet formalism and other parameters specified in a configuration YAML file. This code is also parallelized for efficient computation.

#### Usage

```bash
mpirun -n <number_of_cores> python3 search_node.py -c <config.yaml> -o <output_directory>
```

- `-c`: Path to the configuration file (`config_V.yaml` or `config_W.yaml`) that defines the parameters for the search.
- `-o`: Path to the output directory where results will be stored.

### `floquet_functions.py`
This module contains helper functions used by both `bulk_bands.py` and `search_node.py`. It is not meant to be executed directly but imported in the other scripts.

## Configuration Files

The YAML configuration file (e.g., `config_V.yaml`) used with `search_node.py` should contain all the necessary parameters for tracking Weyl nodes, such as the k-space grid, Floquet parameters, etc.

## Performance Information

- Running `bulk_bands.py` typically takes a few minutes on a standard desktop computer.
- The `search_node.py` script, which tracks the movement of Weyl nodes, is computationally intensive and can take several days on a regular desktop. It is designed to be run on a computational cluster. For instance, on a cluster with 64 cores, this computation takes approximately 12 hours.

## License

This project is licensed under the MIT License.

## Acknowledgements

Institute for Theoretical Physics and Bremen Center for Computational Materials Science, University of Bremen, Bremen, Germany.

## Contact

For questions or issues, feel free to open an issue or reach out to the project maintainers.
