#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <bit>
#include <bitset>
#include <bits/stdc++.h>
#include <limits.h>

using namespace std;

// build the bit table for the dataset
void build_bit_table(vector<vector<int>> &data, vector<vector<vector<int>>> &bit_table, int size, int snp_size)
{
    // snp_size * 3 * number of samples
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < snp_size; j++)
        {

            bit_table[j][data[i][j]][i] = 1;
        }
    }
}

// build the contingency table from the bit table
void build_contingency_table(vector<vector<vector<int>>> &bit_table, vector<vector<vector<vector<vector<vector<int>>>>>> &contingency_table, int size, int snp_size)
{
    for (int snp0 = 0; snp0 < snp_size - 2; snp0++)
    {
        for (int snp1 = snp0 + 1; snp1 < snp_size - 1; snp1++)
        {
            for (int snp2 = snp1 + 1; snp2 < snp_size; snp2++)
            {
                for (int snp0_type = 0; snp0_type < 3; snp0_type++)
                {
                    for (int snp1_type = 0; snp1_type < 3; snp1_type++)
                    {
                        for (int snp2_type = 0; snp2_type < 3; snp2_type++)
                        {
                            int count = 0;
                            for (int i = 0; i < size; i++)
                            {
                                count += (bit_table[snp0][snp0_type][i] & bit_table[snp1][snp1_type][i] & bit_table[snp2][snp2_type][i]);
                            }
                            contingency_table[snp0][snp1][snp2][snp0_type][snp1_type][snp2_type] = count;
                        }
                    }
                }
            }
        }
    }
}

// calculate k2 score
void k2_score(vector<vector<vector<vector<vector<vector<int>>>>>> &control_contingency_table, vector<vector<vector<vector<vector<vector<int>>>>>> &case_contingency_table, int snp_size)
{
    double k2 = DBL_MAX;
    int final_snp0, final_snp1, final_snp2;
    for (int snp0 = 0; snp0 < snp_size - 2; snp0++)
    {
        for (int snp1 = snp0 + 1; snp1 < snp_size - 1; snp1++)
        {
            for (int snp2 = snp1 + 1; snp2 < snp_size; snp2++)
            {
                double score = 0;
                for (int snp0_type = 0; snp0_type < 3; snp0_type++)
                {
                    for (int snp1_type = 0; snp1_type < 3; snp1_type++)
                    {
                        for (int snp2_type = 0; snp2_type < 3; snp2_type++)
                        {
                            int case_count = case_contingency_table[snp0][snp1][snp2][snp0_type][snp1_type][snp2_type];
                            int control_count = control_contingency_table[snp0][snp1][snp2][snp0_type][snp1_type][snp2_type];
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
                    }
                }
                if (score < k2)
                {
                    k2 = score;
                    final_snp0 = snp0;
                    final_snp1 = snp1;
                    final_snp2 = snp2;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{

    int n_samples = 0;
    int control_size = 0;
    int case_size = 0;
    int snp_size = 0;

    // read the dataset
    fstream fin;
    fin.open(argv[1], ios::in);
    string line, word;
    vector<int> sample;
    vector<vector<int>> control_data;
    vector<vector<int>> case_data;
    while (getline(fin, line, '\n'))
    {
        istringstream s(line);
        while (std::getline(s, word, ','))
        {
            if (word == "X")
            {
                break;
            }
            sample.push_back(stoi(word));
        }
        if (sample.size() > 0)
        {
            if (sample[sample.size() - 1] == 0)
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
            n_samples++;
        }
    }
    fin.close();

    // get the number of snps
    int snp_size = control_data[0].size();

    // initialize the bit table
    vector<vector<vector<int>>> control_bit_table(snp_size, vector<vector<int>>(3, vector<int>(control_size, 0)));
    vector<vector<vector<int>>> case_bit_table(snp_size, vector<vector<int>>(3, vector<int>(case_size, 0)));

    // build the bit table
    build_bit_table(control_data, control_bit_table, control_size, snp_size);
    build_bit_table(case_data, case_bit_table, case_size, snp_size);

    // initialize the contingency table
    vector<vector<vector<vector<vector<vector<int>>>>>> control_contingency_table(snp_size, vector<vector<vector<vector<vector<int>>>>>(snp_size, vector<vector<vector<vector<int>>>>(snp_size, vector<vector<vector<int>>>(3, vector<vector<int>>(3, vector<int>(3, 0))))));
    vector<vector<vector<vector<vector<vector<int>>>>>> case_contingency_table(snp_size, vector<vector<vector<vector<vector<int>>>>>(snp_size, vector<vector<vector<vector<int>>>>(snp_size, vector<vector<vector<int>>>(3, vector<vector<int>>(3, vector<int>(3, 0))))));

    // build the contingency table
    build_contingency_table(control_bit_table, control_contingency_table, control_size, snp_size);
    build_contingency_table(case_bit_table, case_contingency_table, case_size, snp_size);

    // calculate the k2 score
    k2_score(control_contingency_table, case_contingency_table, snp_size);

    return 0;
}