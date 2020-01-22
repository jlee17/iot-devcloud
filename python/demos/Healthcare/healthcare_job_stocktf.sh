
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
stocktf_exist=/data/software/miniconda3/4.7.12/bin/conda info --envs | grep "stocktf "
echo $stocktf_exist

if [ -z "$stocktf_exist" ]
then
    /data/software/miniconda3/4.7.12/bin/conda config --set auto_activate_base false
    /data/software/miniconda3/4.7.12/bin/conda env create -f "stocktf.yml"
fi

# Running the code inside stocktf conda env
SAMPLEPATH=$PBS_O_WORKDIR
source /data/software/miniconda3/4.7.12/etc/profile.d/conda.sh
conda activate stocktf
python3 healthcare_no_openvino.py -r $1
conda deactivate
