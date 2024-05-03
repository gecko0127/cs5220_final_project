#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <bit>
#include <bitset>
#include <bits/stdc++.h>
#include <cmath>
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

using namespace std;

// generate 3 order combinations
void generate_combinations(uint16_t *combinations, int snp_size)
{
    uint16_t i, j, k;
    int count = 0;
    for (i = 0; i < snp_size - 2; i++)
    {
        for (j = i + 1; j < snp_size - 1; j++)
        {
            for (k = j + 1; k < snp_size; k++)
            {
                combinations[count++] = i;
                combinations[count++] = j;
                combinations[count++] = k;
            }
        }
    }
}

// construct the bin count
__global__ void build_contingency_table(uint64_t *bit_table, uint16_t *contingency_table, uint16_t *combinations, int num_sample_64_mutiples, int num_combinations)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_combinations; i += blockDim.x * gridDim.x)
    {
        uint8_t idx, j;
        uint64_t t;
        for (idx = 0; idx < 27; idx++)
        {
            for (j = 0; j < num_sample_64_mutiples; j++)
            {
                t = ((bit_table[combinations[i * 3 + 0] * 3 * num_sample_64_mutiples + (idx / 9) * num_sample_64_mutiples + j]) & (bit_table[combinations[i * 3 + 1] * 3 * num_sample_64_mutiples + ((idx % 9) / 3) * num_sample_64_mutiples + j]) & (bit_table[combinations[i * 3 + 2] * 3 * num_sample_64_mutiples + (idx % 3) * num_sample_64_mutiples + j]));
                contingency_table[i * 27 + idx] += __popcll(t);
            }
        }
    }
}

__global__ void k2_score(uint16_t *control_contingency_table, uint16_t *case_contingency_table, double *scores, int num_combinations)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_combinations; i += blockDim.x * gridDim.x)
    {
        for (int idx = 0; idx < 27; idx++)
        {
            for (int b = 1; b <= case_contingency_table[i * 27 + idx] + control_contingency_table[i * 27 + idx] + 1; b++)
            {
                scores[i] += logf(b);
            }
            for (int d = 1; d <= case_contingency_table[i * 27 + idx]; d++)
            {
                scores[i] -= logf(d);
            }
            for (int d = 1; d <= control_contingency_table[i * 27 + idx]; d++)
            {
                scores[i] -= logf(d);
            }
        }
    }
}

int main(int argc, char *argv[])
{

    // cuda variable setup
    int devId;
    cudaGetDevice(&devId);
    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, devId);
    int device_blks = 32 * numSM;

    uint16_t control_size = atoi(argv[2]), case_size = atoi(argv[2]);
    uint8_t control_64_multiples = ceil(control_size * 1.0 / 64), case_64_multiples = ceil(case_size * 1.0 / 64);
    uint16_t snp_size = atoi(argv[3]);
    int num_combinations = snp_size * (snp_size - 1) * (snp_size - 2) / 6;
    uint64_t *control_bit_table = (uint64_t *)malloc(snp_size * 3 * control_64_multiples * sizeof(uint64_t)); // host copy
    memset(control_bit_table, 0, snp_size * 3 * control_64_multiples * sizeof(uint64_t));
    uint64_t *case_bit_table = (uint64_t *)malloc(snp_size * 3 * case_64_multiples * sizeof(uint64_t)); // host copy
    memset(case_bit_table, 0, snp_size * 3 * case_64_multiples * sizeof(uint64_t));

    // read the dataset
    fstream fin;
    fin.open(argv[1], ios::in);
    string line, word;
    uint16_t sample_count = 0;
    while (getline(fin, line, '\n'))
    {
        istringstream s(line);
        int i = 0;
        while (std::getline(s, word, ','))
        {
            if (word == "X")
            {
                break;
            }
            if (i == snp_size)
            {
                sample_count++;
                break;
            }
            if (sample_count < control_size)
            {
                control_bit_table[i * 3 * control_64_multiples + (word[0] - '0') * control_64_multiples + (sample_count / 64)] |= ((uint64_t)1 << (sample_count % 64));
            }
            else
            {
                case_bit_table[i * 3 * case_64_multiples + (word[0] - '0') * case_64_multiples + ((sample_count - control_size) / 64)] |= ((uint64_t)1 << ((sample_count - control_size) % 64));
            }
            i += 1;
        }
    }
    fin.close();
    auto start_time = std::chrono::steady_clock::now();

    uint16_t *combinations = (uint16_t *)malloc(num_combinations * 3 * sizeof(uint16_t));
    generate_combinations(combinations, snp_size);
    uint16_t *d_combinations; // device copy
    cudaMalloc((void **)&d_combinations, 3 * num_combinations * sizeof(uint16_t));
    cudaMemcpy(d_combinations, combinations, 3 * num_combinations * sizeof(uint16_t), cudaMemcpyHostToDevice);

    uint64_t *d_control_bit_table;
    cudaMalloc((void **)&d_control_bit_table, snp_size * 3 * control_64_multiples * sizeof(uint64_t));
    cudaMemcpy(d_control_bit_table, control_bit_table, snp_size * 3 * control_64_multiples * sizeof(uint64_t), cudaMemcpyHostToDevice); // copy control bit table memory from host to device
    free(control_bit_table);
    uint16_t *d_control_contingency_table;

    // device copy
    cudaMalloc((void **)&d_control_contingency_table, num_combinations * 27 * sizeof(uint16_t)); // allocate memory for control contingency table device copy

    build_contingency_table<<<device_blks, NUM_THREADS>>>(d_control_bit_table, d_control_contingency_table, d_combinations, control_64_multiples, num_combinations);
    cudaFree(d_control_bit_table);

    uint64_t *d_case_bit_table;                                                                                                // allocate memory for control bit table device copy
    cudaMalloc((void **)&d_case_bit_table, snp_size * 3 * case_64_multiples * sizeof(uint64_t));                               // allocate memory for case bit table device copy
    cudaMemcpy(d_case_bit_table, case_bit_table, snp_size * 3 * case_64_multiples * sizeof(uint64_t), cudaMemcpyHostToDevice); // copy case bit table memory from host to device
    free(case_bit_table);
    uint16_t *d_case_contingency_table;                                                       // device copy
    cudaMalloc((void **)&d_case_contingency_table, num_combinations * 27 * sizeof(uint16_t)); // allocate memory for case contingency table device copy
    build_contingency_table<<<device_blks, NUM_THREADS>>>(d_case_bit_table, d_case_contingency_table, d_combinations, case_64_multiples, num_combinations);
    cudaFree(d_case_bit_table);
    cudaFree(d_combinations);

    // calculate the k2 score and return the score and resulting combination
    double *d_scores;                                                  // device copy
    cudaMalloc((void **)&d_scores, sizeof(double) * num_combinations); // allocate scores device memory
    cudaMemset(d_scores, 0, sizeof(double) * num_combinations);        // set scores device memory
    k2_score<<<device_blks, NUM_THREADS>>>(d_control_contingency_table, d_case_contingency_table, d_scores, num_combinations);
    cudaFree(d_control_contingency_table);
    cudaFree(d_case_contingency_table);
    double *scores = (double *)malloc(sizeof(double) * num_combinations); // host copy
    cudaMemcpy(scores, d_scores, sizeof(double) * num_combinations, cudaMemcpyDeviceToHost);
    cudaFree(d_scores);
    int best_idx = 0;
    for (int i = 0; i < num_combinations; i++)
    {
        best_idx = (scores[i] < scores[best_idx]) ? i : best_idx;
    }
    cout << "The lowest K2 score: " << scores[best_idx] << endl;
    cout << "The most likely combination of snps: " << combinations[best_idx * 3 + 0] << " " << combinations[best_idx * 3 + 1] << " " << combinations[best_idx * 3 + 2] << endl;
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    cout << "Finish in " << seconds << " seconds." << endl;

    free(scores);
    free(combinations);
    return 0;
}