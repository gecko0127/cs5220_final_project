# CS5220 Final Project: Parallel Epistasis Detection

## Papers:

- [Tensor-Accelerated Fourth-Order Epistasis Detection on GPUs](https://drive.google.com/drive/u/5/folders/1LThV3SBCF0LTDjzMxtgUwkeyCmK9ZleM)
- [Fast search of third-order epistatic interactions on CPU and GPU clusters](https://journals.sagepub.com/doi/10.1177/1094342019852128)
    - This uses the MPI
- [Accelerating 3-way Epistasis Detection with CPU+GPU processing](https://jsspp.org/papers20/ricardo_nobre-epistasis.pdf)
- [GPU-accelerated exhaustive search for third-order epistatic interactions in case–control studies](https://www.sciencedirect.com/science/article/pii/S1877750315000393)
- [Retargeting Tensor Accelerators for Epistasis Detection](https://ieeexplore.ieee.org/document/9357942)


## Github:
- https://github.com/hiperbio/tensor-episdet
- https://github.com/rjfnobre/crossarch-episdet/tree/main
- https://github.com/UDC-GAC/mpi3snp


## Data:
- Hard to find real data; additional request required: https://adni.loni.usc.edu/data-samples/access-data/
- Use dataset for simulation from https://github.com/rjfnobre/crossarch-episdet/tree/main

## Serial Implementation in the main.cpp

### Data Structure
- readin data: `control_data`, `case_data`
    - dimension: (number of samples) * (number of snps)
    - data representation: `vector<vector<char>>`
- combinations: `combinations`
    - dimension: (number of possible combinations (snp_size choose 3)) * 3
    - data representation: `vector<vector<int>>`
- bit table: `control_bit_table`, `case_bit_table`
    - dimension: (number of snps) * (number of genotype per snp (3)) * (number of 64 multiple in the sample (ceiling))
    - data representation: `vector<vector<vector<bitset<64>>>>`
        - I use bitset because each number in the bit table can be actually be stored with a <b>bit</b> size in the memory
        - 64 because:
            1. the size of bitset needs to be a constant not a variable
            2. in the x64 architectures, bitset will be stored as multiples of 64 bits (same as the unsigned long long)
        - more about bitset: https://en.cppreference.com/w/cpp/utility/bitset
- contingency table: `control_contingency_table`, `control_contingency_table`
    - dimension: (number of combinations) * (number of genotype combinations (3 * 3 * 3))
    - data representation: `vector<vector<int>>`
