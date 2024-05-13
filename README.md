# CS5220 Final Project: Parallel Epistasis Detection

## Implementation and Data

### Data

The datasets used in this project are in the [dataset](https://github.com/gecko0127/cs5220_final_project/tree/master/dataset) folder. The data files are generated with the [data_generator.py](https://github.com/gecko0127/cs5220_final_project/blob/master/data_generator.py) due to the confidentiality issue of getting the real data.

### Implementation

Here is a list of implementations discussed in the report. The original copy and implementation history can be found in the link at the end of each implementation method.

- Serial Implementation: `serial.cpp` ([master/serial.cpp](https://github.com/gecko0127/cs5220_final_project/blob/master/serial.cpp))
- OpenMP Implementation: `openmp.cpp` ([ezgi/cs5220_final_project/openmp.cpp](https://github.com/gecko0127/cs5220_final_project/blob/ezgi/cs5220_final_project/openmp.cpp))
- MPI Implementation: `mpi_main.cpp` ([yiwent/mpi_main.cpp](https://github.com/gecko0127/cs5220_final_project/blob/yiwent/mpi_main.cpp))
- Hybrid Implementation: `mpi_openmp_main.cpp` ([yiwent/mpi_openmp_main.cpp](https://github.com/gecko0127/cs5220_final_project/blob/yiwent/mpi_openmp_main.cpp))
- Initial GPU Implementaion: `initial_gpu.cu` ([cris/main.cu](https://github.com/gecko0127/cs5220_final_project/blob/cris/main.cu))
- Optimized GPU Implementation: `optimized_gpu.cu` ([yiwent/main.cu](https://github.com/gecko0127/cs5220_final_project/blob/yiwent/main.cu))

## Reference

### Papers

- [Tensor-Accelerated Fourth-Order Epistasis Detection on GPUs](https://drive.google.com/drive/u/5/folders/1LThV3SBCF0LTDjzMxtgUwkeyCmK9ZleM)
- [Fast search of third-order epistatic interactions on CPU and GPU clusters](https://journals.sagepub.com/doi/10.1177/1094342019852128)
  - This uses the MPI
- [Accelerating 3-way Epistasis Detection with CPU+GPU processing](https://jsspp.org/papers20/ricardo_nobre-epistasis.pdf)
- [GPU-accelerated exhaustive search for third-order epistatic interactions in case–control studies](https://www.sciencedirect.com/science/article/pii/S1877750315000393)
- [Retargeting Tensor Accelerators for Epistasis Detection](https://ieeexplore.ieee.org/document/9357942)

### Github

- <https://github.com/hiperbio/tensor-episdet>
- <https://github.com/rjfnobre/crossarch-episdet/tree/main>
- <https://github.com/UDC-GAC/mpi3snp>

### Data

- Hard to find real data; additional request required: <https://adni.loni.usc.edu/data-samples/access-data/>
- Use dataset for simulation from <https://github.com/rjfnobre/crossarch-episdet/tree/main>
