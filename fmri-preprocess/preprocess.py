#!/usr/bin/env python3

import nibabel as nib
import os
import h5py
from sklearn import linear_model
from scipy import stats
import numpy as np
import json

#path to data
fmripreppath = '../fmriprep/derivatives/'

#path to output
prepath = '../preprocessed/'

#list of sessions
sesslist = ['ses-wk1','ses-wk2','ses-wk3','ses-wk4','ses-wk5','ses-wk6']

#list of tasks
#put your tasks here
tasklist = [['vid1', 'vid2', 'vid3', 'vid4', 'vid5'],
            ['vid1', 'vid2', 'vid3'],
            ['vid1', 'vid2', 'vid3', 'vid4'],
            ['vid1', 'vid2', 'vid3', 'vid4'],
            ['vid1', 'vid2', 'vid3', 'vid4', 'vid5'],
            ['placement', 'wk1recap', 'wk2recap', 'wk3recap', 'wk4recap', 'wk5recap']]

#list of subjects
# sub-s102 1) ses-wk6 wk4recap no cosine01
# sub-s103 1) no ses-wk3 2) ses-wk6 wk4recap no cosine01
# sub-s105 1) ses-wk6 wk4recap no cosine01
# sub-s106 1) ses-wk6 wk4recap no cosine01
# after sub-s106 w4recap, w5recap was skipped
# sub-s112 1) no ses-wk6 placement
# sub-s201 1) no ses-wk6 placement

subs= ['sub-s102','sub-s103','sub-s105','sub-s106','sub-s107',
       'sub-s108','sub-s110','sub-s111','sub-s112','sub-s113',
       'sub-s114','sub-s114','sub-s116','sub-s118','sub-s120',
       'sub-s121','sub-s122','sub-s125','sub-s126','sub-s127','sub-s129','sub-s201','sub-s213','sub-s214','sub-s215','sub-s216']
experts = ['sub-s201','sub-s213','sub-s214','sub-s215','sub-s216']

for sub in subs:
    print('Processing subject:', sub)

    for sess_ind, sess in enumerate(sesslist):
        print('session:', sess)

        if np.isin(sub, experts) and sess != 'ses-wk6':
            continue

        for task_name in tasklist[sess_ind]:
            print('task:', task_name)

            D = dict()

            for hem in ['L', 'R', 'Vol']:
                if hem == 'Vol':
                    fname = os.path.join(fmripreppath + sub + '/' + sess + '/func/' + \
                                         sub + '_' + sess + '_task-' + task_name + '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                    print('      Loading ', fname)
                    nii_4d = nib.load(fname).get_fdata()
                    mask_fname = os.path.join(fmripreppath + sub + '/' + sess + '/func/' + \
                                              sub + '_' + sess + '_task-' + task_name + '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
                    mask_3d = nib.load(mask_fname).get_fdata().astype(bool)
                    D[hem] = nii_4d[mask_3d] # outputs a voxels x time
                else:
                    fname = os.path.join(fmripreppath + sub + '/' + sess + '/func/' + \
                        sub + '_' + sess + '_task-' + task_name + '_hemi-' + hem +'_space-fsaverage6_bold.func.gii')
                    print('      Loading ', fname)

                    gi = nib.load(fname)
                    D[hem] = np.column_stack([gi.darrays[t].data for t in range(len(gi.darrays))])

        # Use regressors for:
        # -CSF
        # -WhiteMatter
        # -FramewiseDisplacement
        # -All cosine bases for drift (0.008 Hz = 125s)
        # -X, Y, Z and derivatives
        # -RotX, RotY, RotZ and derivatives
            conf = np.genfromtxt(os.path.join(fmripreppath + sub + '/' + sess + '/func/' + \
                                              sub + '_' + sess + '_task-' + task_name + '_desc-confounds_timeseries.tsv'),
                                 names=True)

            conf_json = json.load(open(os.path.join(fmripreppath + sub + '/' + sess + '/func/' + \
                                          sub + '_' + sess + '_task-' + task_name + '_desc-confounds_timeseries.json')))
            first_cc = 0
            while True:
                if conf_json['a_comp_cor_%02d' % first_cc]['Mask'] == 'combined':
                    break
                first_cc += 1

            ## Adjust the regressors here #####################
            reg = np.column_stack((conf['trans_x'],
                                   conf['trans_x_derivative1'],
                                   conf['trans_y'],
                                   conf['trans_y_derivative1'],
                                   conf['trans_z'],
                                   conf['trans_z_derivative1'],
                                   conf['rot_x'],
                                   conf['rot_x_derivative1'],
                                   conf['rot_y'],
                                   conf['rot_y_derivative1'],
                                   conf['rot_z'],
                                   conf['rot_z_derivative1'],

                                   # the first six components
                                   conf['a_comp_cor_%02d' % first_cc],
                                   conf['a_comp_cor_%02d' % (first_cc+1)],
                                   conf['a_comp_cor_%02d' % (first_cc+2)],
                                   conf['a_comp_cor_%02d' % (first_cc+3)],
                                   conf['a_comp_cor_%02d' % (first_cc+4)],
                                   conf['a_comp_cor_%02d' % (first_cc+5)],

                                   conf['framewise_displacement'],
                                   conf['cosine00'],
                                   conf['cosine01']))

            #####################################################
            reg = np.nan_to_num(reg)
            print('      Cleaning and zscoring')
            for hem in ['L', 'R', 'Vol']:
                regr = linear_model.LinearRegression()
                regr.fit(reg, D[hem].T)
                D[hem] = D[hem] - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]
                # Note 8% of values on cortical surface are NaNs, and the following will therefore throw an error
                D[hem] = stats.zscore(D[hem], axis=1)

            # Save hdf5 file
            with h5py.File(os.path.join(prepath + sub + '_' + sess + '_' + task_name +  '.h5'),'w') as hf:
                grp = hf.create_group(task_name)
                grp.create_dataset('L', data=D['L'])
                grp.create_dataset('R', data=D['R'])
                grp.create_dataset('Vol', data=D['Vol'])
                grp.create_dataset('reg',data=reg)

            template = nib.load(os.path.join(fmripreppath + sub + '/' + sess + '/func/' + \
                                             sub + '_' + sess + '_task-' + task_name + '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
            Vol_4d = np.zeros(nii_4d.shape)
            Vol_4d[mask_3d] = D['Vol']
            new_img = nib.Nifti1Image(Vol_4d, template.affine, template.header)
            nib.save(new_img, os.path.join(prepath + sub + '_' + sess + '_' + task_name +  '.nii.gz'))
            print('      saved hdf5 file')

