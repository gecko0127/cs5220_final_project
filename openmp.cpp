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

#include <omp.h>

using namespace std;

// generate 3 order combinations
void generate_combinations(vector<vector<char>> &combinations, int snp_size)
{
    #pragma omp for collapse(3)
    // insert elements statically
    for (int i = 0; i < snp_size - 2; i++)
    {
        for (int j = i + 1; j < snp_size - 1; j++)
        {
            for (int k = j + 1; k < snp_size; k++)
            {
		// critical section
                combinations[] = {char(i + '0'), char(j + '0'), char(k + '0')};
                
            }
        }
    }
}

// build the bit table for the dataset
void build_bit_table(vector<vector<char>> &data, vector<vector<vector<bitset<64>>>> &bit_table, int size, int snp_size)
{
    #pragma omp for collapse(2)
    // change loop order 
    for (int j = 0; j < size; j++)
    {
        for (int i = 0; i < snp_size; i++)
        {
            int x = j / 64;
            int y = j % 64;
            bit_table[i][data[j][i] - '0'][x][y] = 1;
        }
    }
}

// build the contingency table from the bit table
void build_contingency_table(vector<vector<vector<bitset<64>>>> &bit_table, vector<vector<int>> &contingency_table, vector<vector<char>> &combinations, int size, int snp_size)
{
    #pragma omp for
    for (int i = 0; i < combinations.size(); i++)
    {
        int snp0 = combinations[i][0] - '0';
        int snp1 = combinations[i][1] - '0';
        int snp2 = combinations[i][2] - '0';
	#pragma omp for
        for (int idx = 0; idx < 27; idx++)
        {
            int snp0_type = idx / 9;
            int snp1_type = (idx % 9) / 3;
            int snp2_type = idx % 3;
            int count = 0;
            for (int i = 0; i < bit_table[snp0][0].size(); i++)
            {
                count += (bit_table[snp0][snp0_type][i] & bit_table[snp1][snp1_type][i] & bit_table[snp2][snp2_type][i]).count();
            }
            contingency_table[i][idx] = count;
        }
    }
}

// calculate k2 score
pair<vector<char>, double> k2_score(vector<vector<int>> &control_contingency_table, vector<vector<int>> &case_contingency_table, int snp_size, vector<vector<char>> &combinations)
{
    double k2 = DBL_MAX;
    vector<char> final_snp;
    #pragma omp for
    for (int i = 0; i < combinations.size(); i++)
    {
        double score = 0;
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
            score += (first_log - second_log);
        }

        if (score < k2)
        {
            k2 = score;
            final_snp = combinations[i];
        }
    }
    return {final_snp, k2};
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
    snp_size = control_data[0].size();

    // generate 3 order combinations (each row is a combination)

    // (number of combinations) * 3
    int num_comb = (snp_size * (snp_size-1) * (snp_size-2)) / 6;
    vector<vector<char>> combinations(comb_size, vector<char>(3));

    generate_combinations(combinations, snp_size);

    // initialize the bit table
    // dimension: snp_size * 3 * (number of 64 multiple in the sample (ceiling))
    vector<vector<vector<bitset<64>>>> control_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(ceil(control_size * 1.0 / 64), 0)));
    vector<vector<vector<bitset<64>>>> case_bit_table(snp_size, vector<vector<bitset<64>>>(3, vector<bitset<64>>(ceil(case_size * 1.0 / 64), 0)));

    // build the bit table
    build_bit_table(control_data, control_bit_table, control_size, snp_size);
    build_bit_table(case_data, case_bit_table, case_size, snp_size);

    // initialize the contingency table
    // (number of combinations) * (number of genotype combinations: 3 * 3 * 3)
    vector<vector<int>> control_contingency_table(combinations.size(), vector<int>(27, 0));
    vector<vector<int>> case_contingency_table(combinations.size(), vector<int>(27, 0));

    // build the contingency table
    build_contingency_table(control_bit_table, control_contingency_table, combinations, control_size, snp_size);
    build_contingency_table(case_bit_table, case_contingency_table, combinations, case_size, snp_size);

    // calculate the k2 score and return the score and resulting combination
    pair<vector<char>, double> result = k2_score(control_contingency_table, case_contingency_table, snp_size, combinations);
    cout << "The lowest K2 score: " << result.second << endl;
    cout << "The most likely combination of snps: " << result.first[0] << " " << result.first[1] << " " << result.first[2] << endl;

    return 0;
}
