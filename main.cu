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
#include <set>

#define NUM_THREADS 256
#define KERNEL_NUM_COMBINATION 1

using namespace std;

void read_in_data(vector<vector<char>> &control_data, vector<vector<char>> &case_data, char *filename)
{
    fstream fin;
    fin.open(filename, ios::in);
    string line, word;
    vector<char> sample;
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
            }
            else
            {
                sample.pop_back();
                case_data.push_back(sample);
            }
            sample.clear();
        }
    }
    fin.close();
}

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

// build the contingency table (CPU function)

void build_contingency_table_cpu(vector<vector<vector<bitset<64>>>> &bit_table, int *local_combinations, int *h_contingency_table, int num_combinations, int snp_size, int num_64_multiples, int device_blks)
{
    // variables and host copy
    int bit_table_size = snp_size * 3 * num_64_multiples * sizeof(uint64_t);
    int contingency_table_size = num_combinations * 27 * sizeof(int);
    uint64_t *h_bit_table = (uint64_t *)malloc(bit_table_size); // host copy of control bit table

    // build a 1d representation of bit table

    for (int i = 0; i < snp_size; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < num_64_multiples; k++)
            {
                int index = i * snp_size * 3 + j * 3 + k;
                h_bit_table[index] = bit_table[i][j][k].to_ullong();
            }
        }
    }

    uint64_t *d_bit_table;    // device copy of bit table
    int *d_contingency_table; // device copy of contingency table

    cudaMalloc((void **)&d_contingency_table, KERNEL_NUM_COMBINATION * 27 * sizeof(int));
    cudaMalloc((void **)d_bit_table, 3 * 3 * sizeof(uint64_t)); // TODO: update for more than one combination
    for (int c = 0; c < num_combinations; c += KERNEL_NUM_COMBINATION)
    {
        int snp0 = local_combinations[c * 3 + 0];
        int snp1 = local_combinations[c * 3 + 1];
        int snp2 = local_combinations[c * 3 + 2];
        cudaMemcpy(d_bit_table, h_bit_table + snp0 * 3 * num_64_multiples, 3 * num_64_multiples * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bit_table + 1 * 3 * num_64_multiples, h_bit_table + snp1 * 3 * num_64_multiples, 3 * num_64_multiples * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bit_table + 2 * 3 * num_64_multiples, h_bit_table + snp2 * 3 * num_64_multiples, 3 * num_64_multiples * sizeof(uint64_t), cudaMemcpyHostToDevice);
        build_contingency_table<<<device_blks, NUM_THREADS>>>(d_bit_table, d_contingency_table, num_64_multiples);
        cudaMemcpy(h_contingency_table + c * 27, d_contingency_table, KERNEL_NUM_COMBINATION * 27 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    free(h_bit_table);
    cudaFree(d_bit_table);
    cudaFree(d_contingency_table);
}

// build the contingency table (GPU kernel)
__global__ void build_contingency_table(uint64_t *bit_table, int *contingency_table, int num_64_multiples)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < 27; idx += stride)
    {
        int snp0_type = idx / 9;
        int snp1_type = (idx % 9) / 3;
        int snp2_type = idx % 3;
        int count = 0;
        for (int m = 0; m < num_64_multiples; m++)
        {
            uint64_t t0 = bit_table[snp0_type * num_64_multiples + m];
            uint64_t t1 = bit_table[1 * 3 * num_64_multiples + snp1_type * num_64_multiples + m];
            uint64_t t2 = bit_table[2 * 3 * num_64_multiples + snp2_type * num_64_multiples + m];
            count + __popcll(t0 & t1 & t2);
        }
        contingency_table[idx] = count;
    }
}

void k2_score_cpu(int *control_contingency_table, int *case_contingency_table, double *scores, int num_combinations, int device_blks)
{
    double *d_scores;
    int *d_control_contingency_table;
    int *d_case_contingency_table;
    int d_num_combinations;
    cudaMalloc((void **)&d_scores, sizeof(double) * d_num_combinations);
    cudaMalloc((void **)&d_control_contingency_table, d_num_combinations * 27 * sizeof(int));
    cudaMalloc((void **)&d_case_contingency_table, d_num_combinations * 27 * sizeof(int));

    for (int c = 0; c < num_combinations; c += d_num_combinations)
    {
        int real_num_combinations = min(d_num_combinations, num_combinations - c);
        cudaMemcpy(d_control_contingency_table, control_contingency_table + c * 27, real_num_combinations * 27 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_case_contingency_table, case_contingency_table + c * 27, real_num_combinations * 27 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_scores, 0, real_num_combinations * sizeof(double));
        k2_score<<<device_blks, NUM_THREADS>>>(d_control_contingency_table, d_case_contingency_table, d_scores, real_num_combinations);
        cudaMemcpy(scores + c, d_scores, sizeof(double) * real_num_combinations, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_scores);
    cudaFree(d_control_contingency_table);
    cudaFree(d_case_contingency_table);
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

    // other vairables
    int control_size = 0;
    int case_size = 0;
    int control_64_multiples = 0;
    int case_64_multiples = 0;
    int snp_size = 0;
    int num_combinations = 0;

    // read the dataset
    // (number of control samples) * snp_size
    vector<vector<char>> control_data;
    vector<vector<char>> case_data;

    read_in_data(control_data, case_data, argv[1]);

    auto start_time = std::chrono::steady_clock::now();
    case_size = case_data.size();
    control_size = control_data.size();
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

    // clean up to save memory
    control_data.clear();
    case_data.clear();

    // CUDA Implementation

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Build contingency table

    // Common
    int *local_combinations = &combinations[0]; // local_copy
    int control_contingency_table_size = num_combinations * 27 * sizeof(int);
    int *control_contingency_table = (int *)malloc(control_contingency_table_size);
    build_contingency_table_cpu(control_bit_table, local_combinations, control_contingency_table, num_combinations, snp_size, control_64_multiples, device_blks);
    control_bit_table.clear();

    int case_contingency_table_size = num_combinations * 27 * sizeof(int);
    int *case_contingency_table = (int *)malloc(case_contingency_table_size);
    build_contingency_table_cpu(case_bit_table, local_combinations, case_contingency_table, num_combinations, snp_size, control_64_multiples, device_blks);
    case_bit_table.clear();

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Calculate k2 score

    double *scores = (double *)malloc(sizeof(double) * num_combinations);

    k2_score_cpu(control_contingency_table, case_contingency_table, scores, num_combinations, device_blks);

    int best_idx = 0;
    for (int i = 0; i < num_combinations; i++)
    {
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