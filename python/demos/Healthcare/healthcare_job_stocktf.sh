cd $PBS_O_WORKDIR


# Running the code inside stocktf conda env
SAMPLEPATH=$PBS_O_WORKDIR
source /data/software/miniconda3/4.7.12/etc/profile.d/conda.sh
conda activate stocktf
python3 healthcare_no_openvino.py -r $1
conda deactivate