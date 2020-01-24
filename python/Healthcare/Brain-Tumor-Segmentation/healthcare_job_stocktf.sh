
# Prevent error and output files from being saved to DevCloud
#PBS -e /dev/null
if [[ $# > 1 ]]; then
    TASKSET="taskset -c $2"
    if [[ $# > 2 ]]; then
        if [[ $3 == "True" ]]; then
            REMOVE_TRAINING_LAYERS="--remove_training_layers"
        fi
    fi
fi
         
cd $PBS_O_WORKDIR 
# Running the code inside stocktf conda env
SAMPLEPATH=$PBS_O_WORKDIR
export PATH=/data/software/miniconda3/4.7.12/miniconda3/condabin:$PATH
source /data/software/miniconda3/4.7.12/etc/profile.d/conda.sh
conda activate stocktf
$TASKSET python3 healthcare_no_openvino.py -r $1 \
                                           --start_index 10 \
                                           --number_iter 1 \
                                           --number_images 80 \
                                           --output_frequency 10 \
                                           $REMOVE_TRAINING_LAYERS 2>&1
conda deactivate
