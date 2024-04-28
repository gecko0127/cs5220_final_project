import csv
import random

def generate_data_file(num_snp, num_sample):
    snps = [0, 1, 2]
    file_name = f"{num_snp}SNPs_{num_sample}samples.csv"
    fields = []
    for i in range(num_snp):
        fields.append("X")
    fields.append("Y")

    rows = []

    for i in range(num_sample):
        row = []
        for j in range(num_snp):
            row.append(random.choice(snps))
        if i < num_sample / 2:
            row.append(0)
        else:
            row.append(1)
        rows.append(row)

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)
    return file_name

def main():
    num_snp = 64
    num_sample = 256
    generate_data_file(num_snp, num_sample)
    print("Data file generated successfully!")
    return 0

if __name__ == "__main__":
    main()