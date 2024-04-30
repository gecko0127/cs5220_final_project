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
#include <chrono>
#include <cuda_runtime.h>



using namespace std;


// generate 3 order combinations
void generate_combinations(vector<vector<int>>& combinations, int snp_size) {
    for (int i = 0; i < snp_size - 2; i++) {
        for (int j = i + 1; j < snp_size - 1; j++) {
            for (int k = j + 1; k < snp_size; k++) {
                vector<int> combination = {i, j, k};
                combinations.push_back(combination);
            }
        }
    }
}

void build_bit_table(const vector<vector<char>>& data, vector<uint64_t>& bit_table, int size, int snp_size) {
    int num_blocks = ceil(size / 64.0);
    bit_table.resize(snp_size * num_blocks, 0);
    for (int i = 0; i < snp_size; ++i) {
        for (int j = 0; j < size; ++j) {
            int allele = data[j][i] - '0';
            int index = i * num_blocks + j / 64;
            int bit_position = j % 64;
            bit_table[index] |= (1ULL << bit_position);
        }
    }
}

// build the bit table for the dataset
void build_bit_table_2(vector<vector<char>> &data, vector<vector<vector<bitset<64>>>> &bit_table, int size, int snp_size)
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

// build the contingency table from the bit table
void build_contingency_table(vector<vector<vector<bitset<64>>>> &bit_table, vector<int> &contingency_table, vector<vector<int>> &combinations, int size, int snp_size)
{
    for (int i = 0; i < combinations.size(); i++)
    {   

        int snp0 = combinations[i][0];
        int snp1 = combinations[i][1];
        int snp2 = combinations[i][2];
        for (int idx = 0; idx < 27; idx++)
        {
            int snp0_type = idx / 9;
            int snp1_type = (idx % 9) / 3;
            int snp2_type = idx % 3;
            int count = 0;
            for (int j = 0; j < bit_table[snp0][0].size(); j++)
            {
                count += (bit_table[snp0][snp0_type][j] & bit_table[snp1][snp1_type][j] & bit_table[snp2][snp2_type][j]).count();
            }
            contingency_table[i*27 + idx] = count;
        }
    }
}


__global__ void build_contingency_table_kernel(uint64_t *bit_table, int *contingency_table, int *combinations, int num_combinations, int num_samples, int snp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_combinations) {
        int snp0 = combinations[idx * 3];
        int snp1 = combinations[idx * 3 + 1];
        int snp2 = combinations[idx * 3 + 2];
        for (int i = 0; i < 27; i++) {
            int snp0_type = i / 9;
            int snp1_type = (i % 9) / 3;
            int snp2_type = i % 3;
            int count = 0;
            for (int j = 0; j < ceil(num_samples * 1.0 / 64); j++) {
                int x = j;
                int index_00 = snp0;
                int index_01 = snp0_type;
                int index_02 = x;
                int index0 = index_00 * 3 * ceil(num_samples * 1.0 / 64) + index_01 * ceil(num_samples * 1.0 / 64) + index_02;
                int index_10 = snp1;
                int index_11 = snp1_type;
                int index_12 = x;
                int index1 = index_10 * 3 * ceil(num_samples * 1.0 / 64) + index_11 * ceil(num_samples * 1.0 / 64) + index_12;
                int index_20 = snp2;
                int index_21 = snp2_type;
                int index_22 = x;
                int index2 = index_20 * 3 * ceil(num_samples * 1.0 / 64) + index_21 * ceil(num_samples * 1.0 / 64) + index_22;
                count += (__popcll(bit_table[index0] & bit_table[index1] & bit_table[index2]));
            }
            contingency_table[idx * 27 + i] = count;
        }
    }
}




__global__ void calculate_k2_score(int *d_case_table, int *d_control_table, double *d_scores, int num_combinations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_combinations) {
        double score = 0.0;
        for (int j = 0; j < 27; j++) {
            int case_count = d_case_table[idx * 27 + j];
            int control_count = d_control_table[idx * 27 + j];
            double first_log = 0.0, second_log = 0.0;
            for (int k = 1; k <= case_count + control_count + 1; k++) {
                first_log += logf((double)k);
            }
            for (int k = 1; k <= case_count; k++) {
                second_log += logf((double)k);
            }
            for (int k = 1; k <= control_count; k++) {
                second_log += logf((double)k);
            }
            score += (first_log - second_log);
        }
        d_scores[idx] = score;
    }
}


int main(int argc, char *argv[])
{
    int control_size = 0;
    int case_size = 0;
    int snp_size = 0;
    bool debug = false; // TODO: set to false if do not want to print out debug info

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
    // get the number of snps
    auto start = std::chrono::high_resolution_clock::now();


    snp_size = control_data[0].size();

    
    vector<vector<int>> combinations;
    generate_combinations(combinations, snp_size);
    //print snp_size
    cout << "snp_size: " << snp_size << endl;


    vector<vector<vector<bitset<64>>>> control_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(ceil(control_size * 1.0 / 64), 0)));
    vector<vector<vector<bitset<64>>>> case_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(ceil(case_size * 1.0 / 64), 0)));

    // build the bit table
    build_bit_table_2(control_data, control_bit_table, control_size, snp_size);
    build_bit_table_2(case_data, case_bit_table, case_size, snp_size);


    vector<uint64_t> flat_control_bit_table;
    for (int i = 0; i < snp_size; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < ceil(control_size * 1.0 / 64); k++) {
                uint64_t value = control_bit_table[i][j][k].to_ullong();
                flat_control_bit_table.push_back(value);
            }
        }
    }

    vector<uint64_t> flat_case_bit_table;
    for (int i = 0; i < snp_size; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < ceil(case_size * 1.0 / 64); k++) {
                uint64_t value = case_bit_table[i][j][k].to_ullong();
                flat_case_bit_table.push_back(value);
            }
        }
    }


    uint64_t* d_case_bit_table;
    size_t size1 = flat_case_bit_table.size() * sizeof(uint64_t);
    cudaMalloc((void**)&d_case_bit_table, size1);
    cudaMemcpy(d_case_bit_table, flat_case_bit_table.data(), size1, cudaMemcpyHostToDevice);

    uint64_t* d_control_bit_table;
    size_t size2 = flat_control_bit_table.size() * sizeof(uint64_t);
    cudaMalloc((void**)&d_control_bit_table, size2);
    cudaMemcpy(d_control_bit_table, flat_control_bit_table.data(), size2, cudaMemcpyHostToDevice);


    int *d_combinations, *d_contingency_table_control, *d_contingency_table_case;
    cudaMalloc(&d_contingency_table_control, combinations.size() * 27 * sizeof(int));
    cudaMalloc(&d_contingency_table_case, combinations.size() * 27 * sizeof(int));

    cudaMalloc(&d_combinations, combinations.size() * 3 * sizeof(int));

    vector<int> flat_combinations;
    for (int i = 0; i < combinations.size(); i++) {
        for (int j = 0; j < 3; j++) {
            int value = combinations[i][j];
            flat_combinations.push_back(value);
        }
    }

    cudaMemcpy(d_combinations, flat_combinations.data(), combinations.size() * 3 * sizeof(int), cudaMemcpyHostToDevice);

    // Define kernel execution parameters
    int blockSize = 256;
    int numBlocksComb = (combinations.size() * 27 + blockSize - 1) / blockSize;

    build_contingency_table_kernel<<<numBlocksComb, blockSize>>>(d_control_bit_table, d_contingency_table_control, d_combinations, combinations.size(), control_size, snp_size);
    build_contingency_table_kernel<<<numBlocksComb, blockSize>>>(d_case_bit_table, d_contingency_table_case, d_combinations, combinations.size(), case_size, snp_size);

    // Free device memory
    cudaFree(d_case_bit_table);
    cudaFree(d_control_bit_table);
    cudaFree(d_combinations);


    double *d_scores;
    cudaMalloc(&d_scores, combinations.size() * sizeof(double));


    int numBlocks = (combinations.size() + blockSize - 1) / blockSize;

    // Launch the kernel
    calculate_k2_score<<<numBlocks, blockSize>>>(d_contingency_table_case, d_contingency_table_control, d_scores, combinations.size());

    // Copy results back to the host
    vector<double> scores(combinations.size());
    cudaMemcpy(&scores[0], d_scores, combinations.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Find the minimum score and corresponding combination
    double minScore = DBL_MAX;
    vector<int> bestCombination;
    for (int i = 0; i < scores.size(); i++) {
        if (scores[i] < minScore) {
            minScore = scores[i];
            bestCombination = combinations[i];
        }
    }

    // Free device memory
    cudaFree(d_scores);


    cout << "The lowest K2 score: " << minScore << endl;
    cout << "The most likely combination of snps: " << bestCombination[0] <<','<< bestCombination[1]<<','<< bestCombination[2]<< endl;

    std::cout << std::endl;
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time gpu: " << elapsed.count() << " s\n";
    return 0;
}