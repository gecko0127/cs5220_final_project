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
void generate_combinations(vector<int> &combinations, int snp_size)
{
    for (int i = 0; i < snp_size - 2; i++)
    {
        for (int j = i + 1; j < snp_size - 1; j++)
        {
            for (int k = j + 1; k < snp_size; k++)
            {
                combinations.push_back(i);
                combinations.push_back(j);
                combinations.push_back(k);
            }
        }
    }
}

// build the bit table for the dataset
void build_bit_table(vector<vector<char>> &data, vector<vector<vector<bitset<64>>>> &bit_table, int size, int snp_size)
{
    for (int i = 0; i < snp_size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int x = j / 64;
            int y = j % 64;
            bit_table[i][data[j][i] - '0'][x][y] = 1;
        }
    }
}

// construct the bin count
__global__ void build_contingency_table(uint64_t *bit_table, int *contingency_table, int *combinations, int num_combinations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_combinations; i += stride)
    {
        int snp0 = combinations[i * 3 + 0];
        int snp1 = combinations[i * 3 + 1];
        int snp2 = combinations[i * 3 + 2];
        for (int idx = 0; idx < 27; idx++)
        {
            int snp0_type = idx / 9;
            int snp1_type = (idx % 9) / 3;
            int snp2_type = idx % 3;
            uint64_t t0 = bit_table[snp0 * 3 + snp0_type];
            uint64_t t1 = bit_table[snp1 * 3 + snp1_type];
            uint64_t t2 = bit_table[snp2 * 3 + snp2_type];
            uint64_t t = (t0 & t1 & t2);
            contingency_table[i * 27 + idx] += __popcll(t);
        }
    }
}

__global__ void k2_score(int *control_contingency_table, int *case_contingency_table, double *scores, int num_combinations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_combinations; i += stride)
    {
        for (int idx = 0; idx < 27; idx++)
        {
            int case_count = case_contingency_table[i * 27 + idx];
            int control_count = control_contingency_table[i * 27 + idx];
            int total_count = case_count + control_count;
            double first_log = 0, second_log = 0;
            for (int b = 1; b <= total_count + 1; b++)
            {
                first_log += logf(b);
            }
            for (int d = 1; d <= case_count; d++)
            {
                second_log += logf(d);
            }
            for (int d = 1; d <= control_count; d++)
            {
                second_log += logf(d);
            }
            scores[i] += (first_log - second_log);
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

    int control_size = 0;
    int case_size = 0;
    int control_64_multiples = 0;
    int case_64_multiples = 0;
    int snp_size = 0;
    int num_combinations = 0;

    // read the dataset
    fstream fin;
    fin.open(argv[1], ios::in);
    string line, word;
    vector<char> sample;

    // (number of control samples) * snp_size
    vector<vector<char>> control_data;
    // (number of case samples ) * snp_size
    vector<vector<char>> case_data;

    while (getline(fin, line, '\n'))
    {
        istringstream s(line);
        while (std::getline(s, word, ','))
        {
            if (word == "X")
            {
                break;
            }
            sample.push_back(word[0]);
        }
        if (sample.size() > 0)
        {
            if (sample[sample.size() - 1] == '0')
            {
                sample.pop_back();
                control_data.push_back(sample);
                control_size++;
            }
            else
            {
                sample.pop_back();
                case_data.push_back(sample);
                case_size++;
            }
            sample.clear();
        }
    }
    fin.close();
    auto start_time = std::chrono::steady_clock::now();

    snp_size = control_data[0].size();                    // get the number of snps
    control_64_multiples = ceil(control_size * 1.0 / 64); // get the number of 64 multiples in control sample
    case_64_multiples = ceil(case_size * 1.0 / 64);       // get the number of 64 multiples in case sample

    // generate 3 order combinations
    vector<int> combinations;
    generate_combinations(combinations, snp_size);
    num_combinations = combinations.size() / 3;

    // initialize the bit table
    // dimension: snp_size * 3 * (number of 64 multiple in the sample (ceiling))
    vector<vector<vector<bitset<64>>>> control_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(control_64_multiples, 0)));
    vector<vector<vector<bitset<64>>>> case_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(case_64_multiples, 0)));

    // build the bit table
    build_bit_table(control_data, control_bit_table, control_size, snp_size);
    build_bit_table(case_data, case_bit_table, case_size, snp_size);

    control_data.clear();
    case_data.clear();

    // CUDA Implementation

    // Common
    int *d_combinations;                        // device copy
    int *local_combinations = &combinations[0]; // local_copy
    int control_bit_table_size = snp_size * 3 * control_64_multiples * sizeof(uint64_t);
    uint64_t *d_control_bit_table;                                                 // device copy of control bit table
    uint64_t *long_control_bit_table = (uint64_t *)malloc(control_bit_table_size); // host copy of control bit table
    for (int k = 0; k < control_64_multiples; k++)
    {
        for (int i = 0; i < snp_size; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int index = k * snp_size * 3 + i * 3 + j;
                long_control_bit_table[index] = control_bit_table[i][j][k].to_ullong();
            }
        }
    }
    int control_contingency_table_size = num_combinations * 27 * sizeof(int);
    int *d_control_contingency_table; // device copy
    int *control_contingency_table = (int *)malloc(control_contingency_table_size);

    // int chunk = 64;
    cudaMalloc((void **)&d_combinations, num_combinations * 3 * sizeof(int));
    cudaMemcpy(d_combinations, local_combinations, num_combinations * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_control_contingency_table, num_combinations * 27 * sizeof(int));
    cudaMalloc((void **)&d_control_bit_table, snp_size * 3 * sizeof(uint64_t)); // allocate memory for control bit table device copy
                                                                                // cudaMemset(d_control_bit_table, 0, chunk * 27 * sizeof(int));
    for (int sample_chunk = 0; sample_chunk < control_64_multiples; sample_chunk++)
    {
        cudaMemcpy(d_control_bit_table, long_control_bit_table + sample_chunk * snp_size * 3 * sizeof(uint64_t), snp_size * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        build_contingency_table<<<device_blks, NUM_THREADS>>>(d_control_bit_table, d_control_contingency_table, d_combinations, num_combinations);
    }
    cudaFree(d_control_bit_table);
    cudaFree(d_control_contingency_table);
    free(long_control_bit_table);
    control_bit_table.clear();

    ///////////////////////////////////////////////////////////////////////////////////

    // Case Part
    int case_bit_table_size = snp_size * 3 * case_64_multiples * sizeof(uint64_t);
    uint64_t *d_case_bit_table;                                              // device copy of case bit table
    uint64_t *long_case_bit_table = (uint64_t *)malloc(case_bit_table_size); // host copy of case bit table
    for (int k = 0; k < case_64_multiples; k++)
    {
        for (int i = 0; i < snp_size; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int index = k * snp_size * 3 + i * 3 + j;
                long_case_bit_table[index] = case_bit_table[i][j][k].to_ullong();
            }
        }
    }

    // cudaMalloc((void **)&d_case_bit_table, case_bit_table_size);                                    // allocate memory for case bit table device copy
    // cudaMemcpy(d_case_bit_table, long_case_bit_table, case_bit_table_size, cudaMemcpyHostToDevice); // copy case bit table memory from host to device
    int case_contingency_table_size = num_combinations * 27 * sizeof(int);
    int *d_case_contingency_table; // device copy
    int *case_contingency_table = (int *)malloc(case_contingency_table_size);

    cudaMalloc((void **)&d_case_contingency_table, num_combinations * 27 * sizeof(int));
    cudaMalloc((void **)&d_case_bit_table, snp_size * 3 * sizeof(uint64_t));
    for (int sample_chunk = 0; sample_chunk < case_64_multiples; sample_chunk++)
    {
        cudaMemcpy(d_case_bit_table, long_case_bit_table + sample_chunk * snp_size * 3 * sizeof(uint64_t), snp_size * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        build_contingency_table<<<device_blks, NUM_THREADS>>>(d_case_bit_table, d_case_contingency_table, d_combinations, num_combinations);
    }
    cudaFree(d_case_bit_table);
    cudaFree(d_case_contingency_table);
    cudaFree(d_combinations);
    free(long_case_bit_table);
    case_bit_table.clear();

    // calculate the k2 score and return the score and resulting combination
    double *scores = (double *)malloc(sizeof(double) * num_combinations); // host copy
    double *d_scores;                                                     // device copy
    //
    // cudaMemset(d_scores, 0, sizeof(double) * num_combinations);           // set scores device memory
    cudaMalloc((void **)&d_scores, sizeof(double) * num_combinations); // allocate scores device memory
    cudaMalloc((void **)&d_control_contingency_table, num_combinations * 27 * sizeof(int));
    cudaMalloc((void **)&d_case_contingency_table, num_combinations * 27 * sizeof(int));
    // for (int start_idx = 0; start_idx < num_combinations; start_idx += chunk)
    //{
    //     int chunk_size = min(chunk, (num_combinations - start_idx));
    cudaMemcpy(d_control_contingency_table, control_contingency_table, num_combinations * 27 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_case_contingency_table, case_contingency_table, num_combinations * 27 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, num_combinations * sizeof(double));
    k2_score<<<device_blks, NUM_THREADS>>>(d_control_contingency_table, d_case_contingency_table, d_scores, num_combinations);
    cudaMemcpy(scores, d_scores, sizeof(double) * num_combinations, cudaMemcpyDeviceToHost);
    //}
    cudaFree(d_scores);
    cudaFree(d_control_contingency_table);
    cudaFree(d_case_contingency_table);

    int best_idx = 0;
    for (int i = 0; i < num_combinations; i++)
    {
        // cout << scores[i] << endl;
        best_idx = (scores[i] < scores[best_idx]) ? i : best_idx;
    }
    cout << "The lowest K2 score: " << scores[best_idx] << endl;
    cout << "The most likely combination of snps: " << combinations[best_idx * 3 + 0] << " " << combinations[best_idx * 3 + 1] << " " << combinations[best_idx * 3 + 2] << endl;
    free(scores);
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    cout << "Finish in " << seconds << " seconds." << endl;

    return 0;
}