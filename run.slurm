#!/bin/bash
#SBATCH -J glean_256_256
#SBATCH -p a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=hs_christal@sjtu.edu.cn
#SBATCH --output=/dssg/home/acct-seesb/seesb-user1/hs/open-mmlab/mmediting/glean_256_256.slurm
#SBATCH --error=/dssg/home/acct-seesb/seesb-user1/hs/open-mmlab/mmediting/glean_256_256.err

module load miniconda3
source activate open-mmlabpy31

sh /dssg/home/acct-seesb/seesb-user1/hs/open-mmlab/mmediting/train.sh



