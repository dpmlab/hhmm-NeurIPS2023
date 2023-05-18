import numpy as np


def get_ev_reps(Ws, intcps, X, segs, embeddings):

    """
    :param Ws: List of weights from fitted HHMM in D x voxel shape
    :param intcps: List of intercept terms from fitted HHMM
    :param X: List of fMRI subject data in time x voxel shape
    :param segs: List of segmentation probabilities from fitted HHMM (segments_) in time x K events
    :param embeddings: Stimulus embeddings in sentence x embeddings
    :return: List of learned representations in each event

    """

    ev_reps = []

    n_subj = len(segs) - 1

    for subj in range(n_subj):
        # compute event representation of each subject
        ev_reps.append(intcps[subj][:, np.newaxis] + (Ws[subj] @ X[subj].T) @ (segs[subj] / np.sum(segs[subj], axis=0)))

    # compute event representation of the stimulus features
    ev_reps.append(intcps[-1][:, np.newaxis] + (Ws[-1] @ embeddings.T) @ (segs[-1] / np.sum(segs[-1], axis=0)))

    return ev_reps


def get_var(ev_reps, verbose=False):

    """
    :param ev_reps: List of event representations in the shape of subject x D (shared latent dimensions) x Event.
    :param verbose: Default is set to False, but True prints out the fMRI and stimulus-fMRI.
        variance explained (VE) at the end
    :return: 2 float values, one for the fMRI VE and one for the stimulus-fMRI VE

    """


    n_ev = ev_reps.shape[2]

    # First compute the sum of squares for the fMRI subjects across all events
    n_fmri_subj = len(ev_reps) - 1

    allev = np.concatenate([np.array(ev_reps[s][:][:]) for s in range(n_fmri_subj)], axis=1)
    SSE_total_fmri = np.sum((allev - allev.mean(1)[:, np.newaxis]) ** 2)

    # Next, compute the sum of squares for fMRI subjects in each event
    SSE_clust_fmri = 0

    for ev in range(n_ev):
        SSE_clust_fmri += np.sum((ev_reps[:-1, :, ev] - ev_reps[:-1, :, ev].mean(0)) ** 2)


    fmri_ve = 1 - SSE_clust_fmri / SSE_total_fmri


    # Compute the sum of squares for the stimulus, compared to the fMRI subjects, across all events
    all_fmri_ev_mean = allev.mean(1)

    SSE_total_stim = np.sum((ev_reps[-1, :, :] - all_fmri_ev_mean[:, np.newaxis]) ** 2)

    # Compute the sum of squares for the stimulus, compared to the fMRI subjects, in each event
    SSE_clust_stim = 0
    for ev in range(n_ev):
        SSE_clust_stim += np.sum((ev_reps[-1, :, ev] - ev_reps[:-1, :, ev].mean(0)) ** 2)

    # Compute final fMRI-stim VE
    stim_fmri_ve = 1 - SSE_clust_stim / SSE_total_stim

    if verbose:
        print('fMRI VE:', fmri_ve)
        print('fMRI<->stimulus VE:', stim_fmri_ve)

    return fmri_ve, stim_fmri_ve


