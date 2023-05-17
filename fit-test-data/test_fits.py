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

