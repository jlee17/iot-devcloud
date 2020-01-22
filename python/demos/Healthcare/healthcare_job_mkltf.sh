
cd $PBS_O_WORKDIR

mkltf_exist=/data/software/miniconda3/4.7.12/bin/conda info --envs | grep "mkltf "

if [ -z "$mkltf_exist" ]
then
    /data/software/miniconda3/4.7.12/bin/conda config --set auto_activate_base false
    /data/software/miniconda3/4.7.12/bin/conda env create -f "mkltf.yml"
fi

# Running the code inside mkltf conda env
SAMPLEPATH=$PBS_O_WORKDIR
source /data/software/miniconda3/4.7.12/etc/profile.d/conda.sh
conda activate mkltf
python3 healthcare_no_openvino.py -r $1 
conda deactivate
