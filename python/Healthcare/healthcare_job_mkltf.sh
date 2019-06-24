
# Prevent error and output files from being saved to DevCloud
#PBS -e /dev/null
#PBS -o /dev/null

cd $PBS_O_WORKDIR 
# Running the code inside mkltf conda env
SAMPLEPATH=$PBS_O_WORKDIR
export PATH=/glob/supplementary-software/versions/Miniconda/miniconda3/condabin:$PATH
source /glob/supplementary-software/versions/Miniconda/miniconda3/etc/profile.d/conda.sh
conda activate mkltf
python3 healthcare_no_openvino.py -r $1 
conda deactivate
