#!/bin/bash
#SBATCH -e /home/arnoldtl/jobs/errors/job_%J.err
#SBATCH -o /home/arnoldtl/jobs/outputs/job_%J.out
#SBATCH -p compute
#SBATCH --mem 300000
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH -J retrieve_contact_domains

## Define code dir and env
CODE_DIR=/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/serendipity
ENV=umbrella_minimal_arnoldtl_16012024

## Go to the correct directory
cd $CODE_DIR

source ~/.bashrc
## Activate the latest env
conda activate umbrella_minimal_arnoldtl_16012024

python /sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/serendipity/preprocessing/resources/contact_domains/retrieve_contact_domains.py
