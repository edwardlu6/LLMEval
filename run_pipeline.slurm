#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --output=llm_eval_%j.out
#SBATCH --error=llm_eval_%j.err
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Load modules (adjust for your HPC)
module load python/intel/3.8

# Set cache directories
export TRANSFORMERS_CACHE=/scratch/$USER/transformers_cache
export HF_HOME=/scratch/$USER/hf_home
mkdir -p $TRANSFORMERS_CACHE $HF_HOME

# Paths - MODIFY THESE
SCRIPT_DIR=/scratch/gl2576/LLMEval              # Your LLMEval root directory
DATA_DIR=${SCRIPT_DIR}/data/data_unified      # Your data is in data_unified/
OUTPUT_DIR=/scratch/$USER/results/${SLURM_JOB_ID}

cd $SCRIPT_DIR
source venv/bin/activate

# Run pipeline for each domain
for DOMAIN in science medical finance; do
    INPUT_FILE=${DATA_DIR}/unified_${DOMAIN}_dev.jsonl
    
    if [ -f "$INPUT_FILE" ]; then
        echo "Processing $DOMAIN domain..."
        
        python hpc_eval_pipeline.py \
            --input $INPUT_FILE \
            --output_dir ${OUTPUT_DIR}/${DOMAIN} \
            --domain $DOMAIN \
            --models deepseek-7b mistral-7b
        
        echo "Completed $DOMAIN"
    else
        echo "Warning: $INPUT_FILE not found, skipping"
    fi
done

echo "=========================================="
echo "Complete: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
