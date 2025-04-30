#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J forecasts
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=10GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 10GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o outputs_hpc/Output_%J.out 
#BSUB -e outputs_hpc/Output_%J.err 

source ../envs/envs/shred/bin/activate

# here follow the commands you want to execute with input.in as the input file
python -u reconstructions.py