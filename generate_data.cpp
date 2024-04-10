// Libraries
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>
#include <sstream>
#include <omp.h>
using namespace std;

// Macros
#define TABLE_MAX_SIZE 748
#define TABLE_ERROR -0.0810

// Class
class SNP
{
	public:
	unsigned long ***data;							// SNPs data matrix
	int nrows;										// #rows in data matrix - SNPs
	int ncols;										// #columns in data matrix - patients
    void generate_data(int num_pac, int num_snp);	// data generator
	void destroy();									// destructor

};

// Destructor
void SNP::destroy()
{
	int i, j;
	// delete matrix "data"
	for(i = 0; i < nrows; i++)
	{
		for(j = 0; j < 3; j++)
			delete []data[i][j];
		delete []data[i];
	}
	delete []data;
}

// Data Generator
void SNP::generate_data(int num_pac, int num_snp)
{
	int i, j, x, new_j, temp;
	unsigned char **tdata;	
	nrows = num_snp/2;
	ncols = num_pac/2;
	// create matric "tdata"
	tdata = new unsigned char*[nrows];
	for(i = 0; i < nrows; i++)
	{
		tdata[i] = new unsigned char[ncols];
	}
	// fill matrix "tdata"
    srand(100);
    for(j = 0; j < ncols; j++)
	{
        for(i = 0; i < nrows; i++)
		{
            tdata[i][j] = rand() % 3;
		}
	}
	// convert data matrix
	ncols = ceil(1.0 * ncols / 64);
	data = new unsigned long**[nrows];
	for(i = 0; i < nrows; i++)
	{
		data[i] = new unsigned long*[3];
		for(j = 0; j < 3; j++)
			data[i][j] = new unsigned long[ncols]();
	}
	for(i = 0; i < nrows; i++)
	{
		new_j = 0;
		for(j = 0; j + 63 < samplesize; j += 64)
		{
			// loop through the 64 columns
			for(x = 0; x < 64; x++)
			{
				// left shift by 1
				data[i][0][new_j] <<= 1;
				data[i][1][new_j] <<= 1;
				data[i][2][new_j] <<= 1;
				// set appropriate position to 1
				data[i][tdata[i][j + x]][new_j] |= 1;
			}
			// update index
			new_j++;
		}
		// repeat for remainder
		if(j != samplesize)
		{
			for(x = 0; j + x < samplesize; x++)
			{
				data[i][0][new_j] <<= 1;
				data[i][1][new_j] <<= 1;
				data[i][2][new_j] <<= 1;
				data[i][tdata[i][j + x]][new_j] |= 1;
			}
		new_j++;
		}
	// delete matrix "tdata"
	for(i = 0; i < nrows; i++)
		delete []tdata[i];
	delete []tdata;
	// end
}