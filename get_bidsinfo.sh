# Use dcm2niix to get additional info about mri acquisition
# Especially slice timing
mri=$1
out=$2
~/mricron/Resources/dcm2niix -f $out -b y -o ~/Dropbox/Mac/Documents/PhD_Work/AVL/DICOM_FILES/bidsinfo ~/Dropbox/Mac/Documents/PhD_Work/AVL/AVL_FMRI/$mri/05-MEMPRAGE_RMS
