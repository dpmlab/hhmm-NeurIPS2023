#!/bin/bash
#SBATCH --job-name=fmriprep_s216

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=99:00:00


study_dir=/fmri-data-directory
fmriprep_dir=/fmriprep-directory

singularity run --cleanenv --bind $studydir \
	$fmriprep_dir/fmriprep-21.0.1.simg \
	$studydir/ds003233/ \
	$studydir/derivatives/ \
	--nthread 8 --omp-nthread 8 \
	participant --participant-label sub-s216 \
	--fs-license-file $studydir/license.txt \
	--work-dir $studydir/derivatives/work \
	--output-space fsaverage fsaverage6 MNI152NLin2009cAsym
