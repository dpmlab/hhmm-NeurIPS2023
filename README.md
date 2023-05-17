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
> Subset and stack fMRI data, per subject, and stimulus embeddings by video or scan session. fMRI data should be 
>inputted into the model with shape time (TR) by voxel. Stimulus embedding data should be inputted as 
>sentences by embeddings. Only one copy of the stimulus needs to be appended to the input list. 
>
>Next, generate labels for both fMRI data and stimulus embeddings. Labels for fMRI data and stimulus embeddings 
>represent the video or scan session in each timepoint. E.g. fMRI input with data from two videos, where the first 
>video is 3 TRs long and the second 5 TRs should have the following labels: [0, 0, 0, 1, 1, 1, 1, 1]. 
>Stimulus input with 2 sentences from the first video and 4 from the second should have the following 
>labels: [0, 0, 1, 1, 1, 1]. Labels must start at 0, representing the virst video or scan session in the input sequence,
>and cumulatively increment by 1 per additional video.
>
>### Example code 
 
    from event import EventSegment
    X, session_labels = [], [] # X should be a list of subjects
    stim_embeddings, stim_labels = [], []
    X.append(stim_embeddings)
    
    event_length = 15 # average length of events in TRs
    D = 3 # number of shared latent dimensions
    ridge_alpha = 10 # alpha parameter for ridge regression

    es = EventSegment(shared_dim=D, align_features=True, event_length=event_length, ridge_alpha=ridge_alpha)
    fit = es.fit(X=X, video_label_TR=session_labels, video_label_sent=stim_labels)
    
> ### Software versions
> - Python v.3.8.13
> - Scipy v.1.7.3
> - Scikit-learn v.1.0.2
> ### Anaconda environment file
> - conda-envs/hyperhmmenv.yml
>### Code 
> - event.py




