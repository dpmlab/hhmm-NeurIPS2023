from event import EventSegment


##

## Stack fMRI data, per subject, and stimulus embeddings by video or scan session.
    ### fMRI data should be inputted into the model with shape time x voxel.
    ### Stimulus embedding data should be inputted as sentences x embeddings.

## Labels for fMRI data and stimulus embeddings represent the video or scan session in each timepoint.
    ### E.g. fMRI input with data from two videos, where the first video is 3 TRs long and the second 5 TRs
    ### should have the following labels: [0, 0, 0, 1, 1, 1, 1, 1]
    ### Stimulus input with 2 sentences from the first video and 4 from the second should have the following
    ### labels: [0, 0, 1, 1, 1, 1]

X, session_labels = [], [] # X should be a list of subjects
stim_embeddings, stim_labels = [], []

X.append(stim_embeddings)

event_length = 15 # average length of events in TRs
D = 3 # number of shared latent dimensions
ridge_alpha = 10 # alpha parameter for ridge regression

es = EventSegment(shared_dim=D, align_features=True, event_length=event_length, ridge_alpha=ridge_alpha)
fit = es.fit(X=X, video_label_TR=session_labels, video_label_sent=stim_labels)

