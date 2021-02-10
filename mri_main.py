import os
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans as kmeans
import MRI_module as M_m
import matplotlib.pyplot as plt
from numba import jit, cuda

jit(target ="cuda")  

#numberf sub_file and k means
number = 8
k = 4
n_pca =30

# data_path
data_path = "C:/Users/yuval/Desktop/Yariv/coding/final_project/data/DTI-EEG/FA_all_plus/stats"

# change PATH
out_file ='C:/Users/yuval/Desktop/Yariv/coding/final_project/data/DTI-EEG/FA_all_plus/statistic_analysis/'+str(number)
filename_1 = 'pk'
filename_2 = 'ht'

# other paths

FA_filename = os.path.join(data_path+'/', "all_FA.nii.gz") 
MASKS_filename = os.path.join(data_path+'/', "statistic_mask.nii.gz")

pk_feature_file = out_file+'/'+'pk_bow'+'.pickle' 
ht_feature_file = out_file+'/'+'ht_bow'+'.pickle'  

pk_labels_file = out_file+'/'+ 'pk_kmm_labels'+'.pickle'
ht_labels_file = out_file+'/'+ 'ht_kmm_labels'+'.pickle'

output_pk_reconstruct = out_file+'/'+ 'pk_reconstruct'
output_ht_reconstruct = out_file+'/'+ 'ht_reconstruct'


# load FA_subjects
if os.path.isdir(out_file) == False :

    os.mkdir(out_file)

#load statistic map
_stat_masks = M_m.load_mri(MASKS_filename)
_stat_mask = _stat_masks[:,:,:,number]
_mask = _stat_mask > 0

p_scor_img = M_m.load_mri('C:/Users/yuval/Desktop/Yariv/coding/final_project/data/DTI-EEG/FA_all/stats/FA_tfce_p_tstat2.nii.gz')
p_95 = p_scor_img >= 0.95

print('total voliume: '+str(np.sum(_mask) ) )

#groups
_data = M_m.load_mri(FA_filename)

fa_pk = np.average(_data[:,:,:,0:11],-1)
fa_ht = np.average(_data[:,:,:,11:],-1)

plt.figure(figsize = (10,10))
plt.imshow(fa_pk[:,:,70])
plt.imshow(_mask[:,:,70], alpha = 0.5)
plt.show()
#timeline 

# extract features for every group
M_m.extract_features(fa_pk,_stat_mask,out_file,filename_1)
M_m.extract_features(fa_ht,_stat_mask,out_file,filename_2)

#bow

pca_model = M_m.BOW(fa_ht,_mask,out_file,filename_2,n_pca,'fit')
M_m.BOW(fa_pk,_mask,out_file,filename_1,n_pca,'predict',pca_model)

'''
# kmeans model trained on ht predicted on pk
_model = kmeans(k)

current_model = M_m.mri_kmeans(ht_feature_file,out_file,filename_1,_model,k,'fit')
current_model = M_m.mri_kmeans(pk_feature_file,out_file,filename_2,current_model,k,'predict')

# reconstruct from labels
M_m.reconstruct_from_labels(pk_labels_file,out_file,filename_1)
M_m.reconstruct_from_labels(ht_labels_file,out_file,filename_2)

# image difference
M_m.diff_mask_labels(out_file+'/'+ 'pk_reconstruct.pickle',out_file+'/'+ 'ht_reconstruct.pickle',out_file)

# 3D connected componets
M_m.mri_cc(out_file+'/'+ 'label_mask.pickle',out_file)
'''
# show 3D image
M_m.show_3d_img(out_file+'/'+'cc_img.pickle',_stat_mask,p_95)
'''
# show slices of labels

M_m.show_images(out_file+'/'+'cc_img.pickle',p_95)

'''