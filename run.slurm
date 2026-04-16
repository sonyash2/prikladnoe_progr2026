#!/bin/bash
#SBATCH --job-name=lab5
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=2gb
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err

module load intel/mpi4

mpirun -np 4 ./mpi matrix1.txt matrix2.txt

echo "Job finished at $(date)"