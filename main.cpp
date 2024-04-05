#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <bits/stdc++.h>

using namespace std;

int main()
{
    int n_samples = 0;
    int control_size = 0;
    int case_size = 0;
    int snp_size = 0;

    vector<int> sample;
    vector<vector<int>> control_data;
    vector<vector<int>> case_data;

    // count the size
    fstream fin;
    fin.open("./dataset/small.csv", ios::in);
    string line, word;

    while (getline(fin, line, '\n'))
    {
        istringstream s(line);
        while (std::getline(s, word, ','))
        {
            if (word == "X")
            {
                break;
            }
            int num = stoi(word);
            sample.push_back(num);
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
    snp_size = control_data[0].size();

    // cout << "n_samples: " << n_samples << endl;
    // cout << "control_size: " << control_size << endl;
    // cout << "case_size: " << case_size << endl;
    // cout << "snp_size: " << snp_size << endl;

    // build the control bit table
    vector<vector<int>> control_bit_table(3, vector<int>(control_size * snp_size, 0));
    for (int i = 0; i < control_size; i++)
    {
        for (int j = 0; j < snp_size; j++)
        {
            control_bit_table[control_data[i][j]][i + j * control_size] = 1;
        }
    }

    // build the case bit table
    vector<vector<int>> case_bit_table(3, vector<int>(case_size * snp_size, 0));
    for (int i = 0; i < case_size; i++)
    {
        for (int j = 0; j < snp_size; j++)
        {
            case_bit_table[case_data[i][j]][i + j * case_size] = 1;
        }
    }

    // print the bit table
    // cout << "control bit table" << endl;
    /*for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < snp_size; j++)
        {
            cout << "This is snp " << j << " : ";
            for (int k = 0; k < control_size; k++)
            {
                cout << control_bit_table[i][k + j * control_size] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }*/

    /*cout << "case bit table" << endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < snp_size; j++)
        {
            cout << "This is snp " << j << " : ";
            for (int k = 0; k < case_size; k++)
            {
                cout << case_bit_table[i][k + j * case_size] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }*/

    // build the contingency table (third order) for control
    vector<vector<vector<vector<vector<vector<int>>>>>> control_contingency_table(snp_size, vector<vector<vector<vector<vector<int>>>>>(snp_size, vector<vector<vector<vector<int>>>>(snp_size, vector<vector<vector<int>>>(3, vector<vector<int>>(3, vector<int>(3, 0))))));
    for (int snp0 = 0; snp0 < snp_size; snp0++)
    {
        for (int snp1 = 0; snp1 < snp_size; snp1++)
        {
            for (int snp2 = 0; snp2 < snp_size; snp2++)
            {
                // if there are same snps, skip
                if (snp0 == snp1 || snp0 == snp2 || snp1 == snp2)
                {
                    continue;
                }
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            vector<int> snp0_bit_vector(control_size);
                            vector<int> snp1_bit_vector(control_size);
                            vector<int> snp2_bit_vector(control_size);
                            int result = 0;
                            int count_one = 0;
                            for (int idx = 0; idx < control_size; idx++)
                            {
                                snp0_bit_vector[idx] = control_bit_table[i][snp0 * control_size + idx];
                                snp1_bit_vector[idx] = control_bit_table[j][snp1 * control_size + idx];
                                snp2_bit_vector[idx] = control_bit_table[k][snp2 * control_size + idx];
                            }
                            // snp0_bit_vector = copy(control_bit_table[i].begin() + snp0 * control_size, control_bit_table[i].begin() + (snp0 + 1) * control_size, snp0_bit_vector);
                            // snp1_bit_vector = copy(control_bit_table[j].begin() + snp1 * control_size, control_bit_table[j].begin() + (snp1 + 1) * control_size, snp1_bit_vector);
                            // snp2_bit_vector = copy(control_bit_table[k].begin() + snp2 * control_size, control_bit_table[k].begin() + (snp2 + 1) * control_size, snp2_bit_vector);
                            for (int idx = 0; idx < snp0_bit_vector.size(); idx++)
                            {
                                result = (snp0_bit_vector[idx] & snp1_bit_vector[idx] & snp2_bit_vector[idx]);
                                count_one += result;
                            }
                            control_contingency_table[snp0][snp1][snp2][i][j][k] = count_one;
                        }
                    }
                }
            }
        }
    }

    // build the contingency table (third order) for case
    vector<vector<vector<vector<vector<vector<int>>>>>> case_contingency_table(snp_size, vector<vector<vector<vector<vector<int>>>>>(snp_size, vector<vector<vector<vector<int>>>>(snp_size, vector<vector<vector<int>>>(3, vector<vector<int>>(3, vector<int>(3, 0))))));
    for (int snp0 = 0; snp0 < snp_size; snp0++)
    {
        for (int snp1 = 0; snp1 < snp_size; snp1++)
        {
            for (int snp2 = 0; snp2 < snp_size; snp2++)
            {
                // if there are same snps, skip
                if (snp0 == snp1 || snp0 == snp2 || snp1 == snp2)
                {
                    continue;
                }
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            vector<int> snp0_bit_vector(case_size);
                            vector<int> snp1_bit_vector(case_size);
                            vector<int> snp2_bit_vector(case_size);
                            int result = 0;
                            // vector<int> result;
                            int count_one = 0;
                            for (int idx = 0; idx < control_size; idx++)
                            {
                                snp0_bit_vector[idx] = control_bit_table[i][snp0 * control_size + idx];
                                snp1_bit_vector[idx] = control_bit_table[j][snp1 * control_size + idx];
                                snp2_bit_vector[idx] = control_bit_table[k][snp2 * control_size + idx];
                            }
                            for (int idx = 0; idx < snp0_bit_vector.size(); idx++)
                            {
                                result = (snp0_bit_vector[idx] & snp1_bit_vector[idx] & snp2_bit_vector[idx]);
                                count_one += result;
                            }
                            case_contingency_table[snp0][snp1][snp2][i][j][k] = count_one;
                        }
                    }
                }
            }
        }
    }

    // print and check the contingency table
    /*     for (int snp0 = 0; snp0 < snp_size; snp0++)
        {
            for (int snp1 = 0; snp1 < snp_size; snp1++)
            {
                for (int snp2 = 0; snp2 < snp_size; snp2++)
                {
                    // if there are same snps, skip
                    if (snp0 == snp1 || snp0 == snp2 || snp1 == snp2)
                    {
                        continue;
                    }
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            for (int k = 0; k < 3; k++)
                            {
                                cout << "snp0: " << snp0 << " = " << i << endl;
                                cout << "snp1: " << snp1 << " = " << j << endl;
                                cout << "snp2: " << snp2 << " = " << k << endl;
                                cout << "count: " << control_contingency_table[snp0][snp1][snp2][i][j][k] << endl;
                                cout << endl;
                            }
                        }
                    }
                }
            }
        } */

    // calculate k2 score
    vector<vector<vector<double>>> k2_score(snp_size, vector<vector<double>>(snp_size, vector<double>(snp_size, 0)));
    for (int snp0 = 0; snp0 < snp_size; snp0++)
    {
        for (int snp1 = 0; snp1 < snp_size; snp1++)
        {
            for (int snp2 = 0; snp2 < snp_size; snp2++)
            {
                if (snp0 == snp1 || snp0 == snp2 || snp1 == snp2)
                {
                    continue;
                }
                double k2 = 0;
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            int r_i = control_contingency_table[snp0][snp1][snp2][i][j][k] + case_contingency_table[snp0][snp1][snp2][i][j][k] + 1;
                            double first_log = 0;
                            for (int b = 1; b <= r_i; b++)
                            {
                                first_log += log(b);
                            }
                            // control
                            double second_log = 0;
                            int r_i_j = control_contingency_table[snp0][snp1][snp2][i][j][k];
                            for (int d = 1; d <= r_i_j; d++)
                            {
                                second_log += log(d);
                            }
                            r_i_j = case_contingency_table[snp0][snp1][snp2][i][j][k];
                            for (int d = 1; d <= r_i_j; d++)
                            {
                                second_log += log(d);
                            }
                            k2 += first_log - second_log;
                        }
                    }
                }
                k2_score[snp0][snp1][snp2] = k2;
            }
        }
    }

    // find the maximum k2 score
    double max_k2 = 0;
    int max_snp0 = 0;
    int max_snp1 = 0;
    int max_snp2 = 0;

    for (int snp0 = 0; snp0 < snp_size; snp0++)
    {
        for (int snp1 = 0; snp1 < snp_size; snp1++)
        {
            for (int snp2 = 0; snp2 < snp_size; snp2++)
            {
                if (snp0 == snp1 || snp0 == snp2 || snp1 == snp2)
                {
                    continue;
                }
                if (k2_score[snp0][snp1][snp2] > max_k2)
                {
                    max_k2 = k2_score[snp0][snp1][snp2];
                    max_snp0 = snp0;
                    max_snp1 = snp1;
                    max_snp2 = snp2;
                }
            }
        }
    }

    return 0;
}