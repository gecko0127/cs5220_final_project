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
#include <mpi.h>
#include <chrono>
#include <omp.h>

using namespace std;

// generate 2 order combinations
void generate_2nd_combinations(vector<vector<int>> &combinations, int snp_size, int num_procs, int rank)
{
    int counter = 0;
    int target = rank;
    for (int i = 0; i < snp_size - 2; i++)
    {
        for (int j = i + 1; j < snp_size - 1; j++)
        {
            if (counter == target)
            {
                for (int k = j + 1; k < snp_size; k++)
                {
                    combinations.push_back({i, j, k});
                }
                target = target + num_procs;
            }
            counter += 1;
        }
    }
}

// generate 3 order combinations
void generate_3rd_combinations(vector<vector<int>> &combinations, int snp_size, int num_procs, int rank)
{
    int counter = 0;
    int target = rank;
    for (int i = 0; i < snp_size - 2; i++)
    {
        for (int j = i + 1; j < snp_size - 1; j++)
        {
            for (int k = j + 1; k < snp_size; k++)
            {
                if (counter == target)
                {
                    combinations.push_back({i, j, k});
                    target = target + num_procs;
                }
                counter++;
            }
        }
    }
}

// build the contingency table from the bit table (2nd order)
void build_2nd_contingency_table(vector<vector<vector<bitset<64>>>> &bit_table, vector<vector<int>> &contingency_table, vector<vector<int>> &combinations, int size, int snp_size)
{
    int num_sample_bit_set = bit_table[0][0].size();
    // 3 * 3 * (num_sample_bit_set)
    vector<vector<vector<bitset<64>>>> temp(3, vector<vector<bitset<64>>>(3, vector<bitset<64>>(num_sample_bit_set, 0)));

#pragma omp parallel for
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
            for (int j = 0; j < num_sample_bit_set; j++)
            {
                if (i == 0 || (snp0 != combinations[i - 1][0] || snp1 != combinations[i - 1][1]))
                {
                    temp[snp0_type][snp1_type][j] = bit_table[snp0][snp0_type][j] & bit_table[snp1][snp1_type][j];
                }
                count += (temp[snp0_type][snp1_type][j] & bit_table[snp2][snp2_type][j]).count();
            }
            contingency_table[i][idx] = count;
        }
    }
}

// build the contingency table from the bit table (3rd combinations)
void build_3rd_contingency_table(vector<vector<vector<bitset<64>>>> &bit_table, vector<vector<int>> &contingency_table, vector<vector<int>> &combinations, int size, int snp_size)
{
    int num_biset = bit_table[0][0].size();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < combinations.size(); i++)
    {
        for (int idx = 0; idx < 27; idx++)
        {
            int snp0 = combinations[i][0];
            int snp1 = combinations[i][1];
            int snp2 = combinations[i][2];
            int snp0_type = idx / 9;
            int snp1_type = (idx % 9) / 3;
            int snp2_type = idx % 3;
            int count = 0;
            for (int j = 0; j < num_biset; j++)
            {
                count += (bit_table[snp0][snp0_type][j] & bit_table[snp1][snp1_type][j] & bit_table[snp2][snp2_type][j]).count();
            }
            contingency_table[i][idx] = count;
        }
    }
}

// build the bit table for the dataset
void build_bit_table(vector<vector<char>> &data, vector<vector<vector<bitset<64>>>> &bit_table, int size, int snp_size)
{
#pragma omp parallel for collapse(2)
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

// calculate k2 score
pair<vector<int>, double> k2_score(vector<vector<int>> &control_contingency_table, vector<vector<int>> &case_contingency_table, int snp_size, vector<vector<int>> &combinations)
{
    double k2 = DBL_MAX;
    vector<int> final_snp;
    vector<double> scores(combinations.size(), 0);
#pragma omp parallel for
    for (int i = 0; i < combinations.size(); i++)
    {
        for (int idx = 0; idx < 27; idx++)
        {
            int case_count = case_contingency_table[i][idx];
            int control_count = control_contingency_table[i][idx];
            int total_count = case_count + control_count;
            double first_log = 0, second_log = 0;
            for (int b = 1; b <= total_count + 1; b++)
            {
                first_log += log(b);
            }
            for (int d = 1; d <= case_count; d++)
            {
                second_log += log(d);
            }
            for (int d = 1; d <= control_count; d++)
            {
                second_log += log(d);
            }
            scores[i] += (first_log - second_log);
        }
    }
    for (int i = 0; i < scores.size(); i++)
    {
        if (scores[i] < k2)
        {
            k2 = scores[i];
            final_snp = combinations[i];
        }
    }
    return {final_snp, k2};
}

// the expected argument: argv[1]: file name; argv[2]: mode (3 for third order combinations, 2 for 2 order combinations)
int main(int argc, char *argv[])
{

    // Init MPI
    // num_procs is the total number of ranks in execution
    // rank is the rank number of current rank

    int num_procs, rank, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int control_size = 0; // the number of control sample
    int case_size = 0;    // the number of case sample
    int snp_size = 0;     // the number of snp
    bool debug = false;   // TODO: set to false if do not want to print out debug info
    int mode = stoi(argv[2]);

    if (rank == 0 && debug)
    {
        cout << "This is mode: " << mode << endl;
    }

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
            // control sample
            if (sample[sample.size() - 1] == '0')
            {
                sample.pop_back();
                control_data.push_back(sample);
                control_size++;
            }
            else
            { // case sample
                sample.pop_back();
                case_data.push_back(sample);
                case_size++;
            }
            sample.clear();
        }
    }
    fin.close();

    // get the number of snps
    snp_size = control_data[0].size();

    if (debug)
    {
        cout << "first stage: read in data\n"
             << endl;

        cout << "this is control data: " << endl;
        for (int i = 0; i < control_size; i++)
        {
            cout << "sample: ";
            for (int j = 0; j < snp_size; j++)
            {
                cout << control_data[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;

        cout << "this is case data: " << endl;
        for (int i = 0; i < case_size; i++)
        {
            cout << "sample: ";
            for (int j = 0; j < snp_size; j++)
            {
                cout << case_data[i][j] << " ";
            }
            cout << endl;
        }
        cout << "-----------------------------------------------" << endl;
        cout << endl;
    }

    auto start_time = std::chrono::steady_clock::now();

    // generate 3 order combinations (each row is a combination)
    vector<vector<int>> combinations;
    if (mode == 3)
    {
        generate_3rd_combinations(combinations, snp_size, num_procs, rank);
    }
    else
    {
        generate_2nd_combinations(combinations, snp_size, num_procs, rank);
    }

    if (debug)
    {
        cout << "second stage: generate the number of combinations of snps\n"
             << endl;
        for (auto &combination : combinations)
        {
            cout << combination[0] << " " << combination[1] << " " << combination[2] << endl;
        }
        cout << "\nthere are " << combinations.size() << " of 3nd order combinations in total." << endl;

        cout << "-----------------------------------------------------" << endl;
    }

    // initialize the bit table
    // dimension: snp_size * 3 * (number of 64 multiple in the sample (ceiling))
    vector<vector<vector<bitset<64>>>> control_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(ceil(control_size * 1.0 / 64), 0)));
    vector<vector<vector<bitset<64>>>> case_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(ceil(case_size * 1.0 / 64), 0)));

    // build the bit table
    build_bit_table(control_data, control_bit_table, control_size, snp_size);
    build_bit_table(case_data, case_bit_table, case_size, snp_size);

    if (debug)
    {
        cout << "This is the third stage: building bit table\n"
             << endl;

        cout << "This is the control bit table: " << endl;
        for (int i = 0; i < snp_size; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cout << "snp: " << i << "; genotype: " << j << " : ";
                for (auto &c : control_bit_table[i][j])
                {
                    cout << c << " ";
                }
                cout << endl;
            }
        }

        cout << endl;

        cout << "This is the case bit table: " << endl;
        for (int i = 0; i < snp_size; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cout << "snp: " << i << "; genotype: " << j << " : ";
                for (auto &c : case_bit_table[i][j])
                {
                    cout << c << " ";
                }
                cout << endl;
            }
        }
        cout << "-------------------------------------------------------" << endl;
        cout << endl;
    }

    // initialize the contingency table
    // (number of combinations) * (number of genotype combinations: 3 * 3 * 3)
    vector<vector<int>> control_contingency_table(combinations.size(), vector<int>(27, 0));
    vector<vector<int>> case_contingency_table(combinations.size(), vector<int>(27, 0));

    // build the contingency table
    if (mode == 3)
    {
        build_3rd_contingency_table(control_bit_table, control_contingency_table, combinations, control_size, snp_size);
        build_3rd_contingency_table(case_bit_table, case_contingency_table, combinations, case_size, snp_size);
    }
    else
    {
        build_2nd_contingency_table(control_bit_table, control_contingency_table, combinations, control_size, snp_size);
        build_2nd_contingency_table(case_bit_table, case_contingency_table, combinations, case_size, snp_size);
    }

    if (debug)
    {
        cout << "This is the fourth stage: building contingency table\n"
             << endl;

        cout << "This is the control contingency table: " << endl;
        for (int i = 0; i < combinations.size(); i++)
        {
            cout << "snp0: " << combinations[i][0] << "; snp1: " << combinations[i][1] << "; snp2: " << combinations[i][2] << endl;
            for (int idx = 0; idx < 27; idx++)
            {
                int snp0_type = idx / 9;
                int snp1_type = (idx % 9) / 3;
                int snp2_type = idx % 3;
                if (control_contingency_table[i][idx] != 0)
                {
                    cout << "genotype: " << snp0_type << " " << snp1_type << " " << snp2_type << " : " << control_contingency_table[i][idx] << endl;
                }
            }
            cout << endl;
        }

        cout << endl;

        cout << "This is the case contingency table: " << endl;
        for (int i = 0; i < combinations.size(); i++)
        {
            cout << "snp0: " << combinations[i][0] << "; snp1: " << combinations[i][1] << "; snp2: " << combinations[i][2] << endl;
            for (int idx = 0; idx < 27; idx++)
            {
                int snp0_type = idx / 9;
                int snp1_type = (idx % 9) / 3;
                int snp2_type = idx % 3;
                if (case_contingency_table[i][idx] != 0)
                {
                    cout << "genotype: " << snp0_type << " " << snp1_type << " " << snp2_type << " : " << case_contingency_table[i][idx] << endl;
                }
            }
            cout << endl;
        }
    }

    // calculate the k2 score and return the score and resulting combination
    pair<vector<int>, double> result = k2_score(control_contingency_table, case_contingency_table, snp_size, combinations);
    double k2_score = result.second;
    int snp_combination[3] = {result.first[0], result.first[1], result.first[2]};

    // local best k2 score and combination
    if (debug)
    {
        cout << "The lowest K2 score: " << result.second << "; rank: " << rank << endl;
        cout << "The most likely combination of snps: " << result.first[0] << " " << result.first[1] << " " << result.first[2] << "; rank: " << rank << endl;
    }

    // buffer
    double final_k2[num_procs] = {0};
    int final_combinations[num_procs * 3] = {0};

    MPI_Gather(&k2_score, 1, MPI_DOUBLE, final_k2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&snp_combination, 3, MPI_INT, final_combinations, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        int index = 0;
        for (int i = 0; i < num_procs; i++)
        {
            if (final_k2[i] < final_k2[index])
            {
                index = i;
            }
        }
        cout << "The final lowest K2 score: " << final_k2[index] << endl;
        cout << "The final most likely combination of snps: " << final_combinations[index * 3] << " " << final_combinations[index * 3 + 1] << " " << final_combinations[index * 3 + 2] << endl;
    }
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    if (rank == 0)
    {
        cout << "Finish in " << seconds << " seconds." << endl;
    }
    MPI_Finalize();
    return 0;
}