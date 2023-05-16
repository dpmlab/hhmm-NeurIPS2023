import numpy as np
import pickle
import brainiak.eventseg.event



def fit_to_half(X, embeddings, label_idxs, emb_label_idxs, all_n_evs, fit_fname,
                baseline=False):

    '''

    :param X: List of fMRI data in the shape time x voxel
    :param embeddings: Stimulus embeddings in the shape embeddings x sentences
    :param label_idxs: List of TR index for each video in fMRI data
    :param emb_label_idxs: List of TR index for each video in stimulus embedding
    :param all_n_evs: List of the number of events in each video
    :param fit_fname: String, filepath and name of the training fits
    :param baseline: True or false for whether or not this is a baseline fit

    '''


    with open(fit_fname, 'rb') as f:
        split_fit = pickle.load(f)
    f.close()

    ev_fits = []

    for vid, n_ev in enumerate(all_n_evs):

        if baseline:

            d = [X[i][label_idxs[vid]:label_idxs[vid + 1], :] @ np.random.standard_normal(split_fit.Ws[i].shape).T +
                 split_fit.intcp[i] for i in range(len(X))]
            d.append(
                embeddings[emb_label_idxs[vid]:emb_label_idxs[vid + 1], :] @ np.random.permutation(split_fit.Ws[-1].T) +
                split_fit.intcp[-1])

        else:
            d = [X[i][label_idxs[vid]:label_idxs[vid + 1], :] @ split_fit.Ws[i].T + split_fit.intcp[i] for i in
                 range(len(X))]
            d.append(
                embeddings[emb_label_idxs[vid]:emb_label_idxs[vid + 1], :] @ split_fit.Ws[-1].T + split_fit.intcp[-1])



        ev = brainiak.eventseg.event.EventSegment(n_ev)
        ev.fit(d)
        ev_fits.append(ev)

    # Save out ev_fits to whatever file format you choose.


## Here we take the heldout or test dataset and fit the learned projection matrices, W, from the training dataset.


## Stack fMRI data, per subject, and stimulus embeddings by video or scan session.
    ### fMRI data should be inputted into the model with shape time x voxel.
    ### Stimulus embedding data should be inputted as sentences x embeddings.

## Index labels for fMRI data and stimulus embeddings represent the timepoint in which the videos occur.
    ### E.g. fMRI input with data from two videos, where the first video is 3 TRs long and the second 5 TRs
    ### should have the following labels: [0, 3]
    ### Stimulus input with 2 sentences from the first video and 4 from the second should have the following
    ### labels: [0, 2]

X, session_idx = [], []
stim_embeddings, stim_idx = [], []


## all_nevs is a list of the number of events for each video in the test set.
    ### E.g. the first video has 5 events, the second has 7, and the third has 3: [5, 7, 3]
all_nevs = []

## fit_fname is the filepath to your training fits. These files are read in to extract learned Ws before fitting to the
    ### heldout dataset

fit_fname = ''

fit_to_half(X, stim_embeddings, session_idx, stim_idx, all_nevs, fit_fname)

