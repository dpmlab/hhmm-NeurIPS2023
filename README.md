# H-HMM: NeurIPS2023 code release



## fMRI data pre-processing
> All fMRI subject files in the [Meshulam et al. 2021 dataset](https://openneuro.org/datasets/ds003233/versions/1.2.0) 
>were preprocessed using fMRIPrep. fMRIPrep was executed inside of a Singularity image on a
> High-Performance Cluster. An example script, written for SLURM protocols, is included along with other code under
>*fmri-preprocess/fmriprep_script.sh*. Next, fMRIPrep output, such as motion regressors, cosine regressors, 
>details of high-pass filtering, are saved out into output file directories. The output is then used to finish preprocessing 
>the data with a Python script (*fmri-preprocess/preprocess.py*), applying these outputs to the data and saving anatomical and 
>functional files associated with each subject. The scripts and software version detail are available below, as is
>the related Anaconda environment file.
>
> ### Software versions
> - Python v.3.8.3
> - fMRIPrep v.21.0.1
> ### Anaconda environment file
> - conda-envs/fmri-prep.yml
>### Code 
> - fmri-preprocess/
>> - fmriprep_script.sh
>> - preprocess.py

## Fitting H-HMM to training data
> 


