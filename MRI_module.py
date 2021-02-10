import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import DBSCAN as dbscan
from skimage.measure import label as diff_label
from sklearn.decomposition import PCA
from scipy import ndimage
import pickle
import cc3d
from mpl_toolkits.mplot3d import Axes3D
import random


def load_mri(FA_filename):

    img_fa = nib.load(FA_filename)
    data = img_fa.get_fdata()

    return data

def save_mri(outfile,name=None):
    
    if name == None:
        with open(out_file+'/'+'.pickle', 'wb') as f:  
            pickle.dump(raw_data, f)
   
    else:
        with open(out_file+'/'+ name +'.pickle', 'wb') as f:  
            pickle.dump(raw_data, f)
    return


def count_labels(_lables):
    
    list_of_labels = [np.sum(_lables == i) for i in range(max(_lables)+1) ]

    return np.array(list_of_labels)

def open_2im(filename):
    
    with open(filename, 'rb') as f:
        raw_data = pickle.load(f)

    img = np.array(raw_data['image'])
   
    return img 


def extract_features(data,stat_mask,out_file,filename):

    mask_idx = np.where(stat_mask == np.max(stat_mask) )
    mask_idx = np.array(mask_idx).T
    img_shape = data.shape
    com = ndimage.center_of_mass(stat_mask)

    c_x = com[0]
    c_y = com[1]
    c_z = com[2]
    w_x_max = np.max(mask_idx[:,0])
    w_y_max = np.max(mask_idx[:,1])
    w_z_max = np.max(mask_idx[:,2])

    mean_pp =list()
    std_pp = list()
    intensity_pp = list()
    laplace_pp = list()
    laplace_std_pp = list() 
    grad_pp = list()
    grad_std_pp = list()
    x_list = list()
    y_list = list()
    z_list = list()
        

    for i in range(0,mask_idx.shape[0]):
        
        w_x = mask_idx[i,0]
        w_y = mask_idx[i,1]
        w_z = mask_idx[i,2]

        data_window = data[w_x-2:w_x+2,w_y-2:w_y+2,w_z-2:w_z+2]

        w_x_dst = ( w_x-c_x ) /(c_x+w_x_max)
        w_y_dst = ( w_y-c_y ) /(c_y+w_y_max)
        w_z_dst = ( w_z-c_z ) /(c_z+w_z_max)
        x_list.append(w_x_dst)
        y_list.append(w_x_dst)
        z_list.append(w_x_dst)

        mean_pp.append(np.mean(data_window ) )
        std_pp.append(np.std(data_window ) )
        intensity_pp.append(data[w_x,w_y,w_z] )
                
        laplacian = ndimage.gaussian_laplace(data_window, sigma=2,mode='reflect', cval=0.0) 
        grad = ndimage.gaussian_gradient_magnitude(data_window,sigma=2,mode='reflect', cval=0.0)

        laplace_pp.append(np.mean(laplacian))
        laplace_std_pp.append(np.std(laplacian ) )
        
        grad_pp.append(np.mean(grad))
        grad_std_pp.append(np.std(grad ) )
       
                
    feat_array = [mean_pp,std_pp,intensity_pp,laplace_pp,laplace_std_pp,grad_pp,grad_std_pp,x_list,y_list,z_list]
    feat_array = np.array(feat_array,dtype=np.float).T


    #write pickle file
    my_dict = { 'features': feat_array.tolist(),
                'mask_idx': mask_idx.tolist(),
                'image_shape' : img_shape            
    }

    with open(out_file+'/'+filename+'_features'+'.pickle', 'wb') as f:
        pickle.dump(my_dict, f)

    return


def BOW(dti,stat_mask,output_file,filename,n_pca,phase,pca_model = None):
    
    mask_idx = np.where(stat_mask == np.max(stat_mask) )
    mask_idx = np.array(mask_idx).T
    dti_shape = dti.shape
    com = ndimage.center_of_mass(stat_mask)

    c_x = com[0]
    c_y = com[1]
    c_z = com[2]

    w_x_max = np.max(mask_idx[:,0])
    w_y_max = np.max(mask_idx[:,1])
    w_z_max = np.max(mask_idx[:,2])
    w_z_min = np.min(mask_idx[:,2])
    feature_space = []
    outfile_dict = {}

    for i in range(0,mask_idx.shape[0]):
        w_x = mask_idx[i,0]
        w_y = mask_idx[i,1]
        w_z = mask_idx[i,2]
        w_x_dst = ( w_x-c_x ) /(w_x_max-c_x)
        w_y_dst = ( w_y-c_y ) /(w_y_max-c_y)
        w_z_dst = ( w_z-c_z ) /(w_z_max-c_z)
        voxel = [w_x_dst,w_y_dst,w_z_dst]
        
        width = 4
        data_window = dti[w_x-width:w_x+width+1,w_y-width:w_y+width+1,w_z-width:w_z +width+1]
        data_window = data_window.flatten()
        feature_space.append(data_window)
    
    feature_space = np.array(feature_space)
    mean_feat = np.mean(feature_space,axis=0)
    feature_space = feature_space - mean_feat

    if phase == 'fit':
        pca_model = PCA(n_pca)
        pca_model = pca_model.fit(feature_space)

    windows_transform = pca_model.transform(feature_space)
    w_i_t = pca_model.inverse_transform(windows_transform)
    coeff = pca_model.components_

    feature_space +=mean_feat
    w_i_t += mean_feat

    recon_im = np.zeros(feature_space[0].shape)
    recon_im += mean_feat
    recon_im += np.matmul(windows_transform[0],coeff)
    
    feature_im = []

    for i in range(n_pca):
        vec_im = coeff[i,:]
        feature_im.append(vec_im)
    
    fig , ax = plt.subplots(1,3)
    fig.suptitle('patch')
    ax[0].imshow(feature_space[0].reshape([width*2+1,width*2+1,width*2+1])[:,:,2] ) 
    ax[0].set_title('original patch')
    ax[1].imshow(w_i_t[0].reshape([width*2+1,width*2+1,width*2+1])[:,:,2] )
    ax[1].set_title('pca patch reconstruction')
    ax[2].imshow(recon_im.reshape([width*2+1,width*2+1,width*2+1])[:,:,2])
    ax[2].set_title('manual patch reconstruction')
    
    fig_1,ax_1 = plt.subplots(3,np.int(np.round(n_pca/3) ) )
    fig_1.set_figheight(20)
    fig_1.set_figwidth(20)
    fig.suptitle('features')
    counter = 0 
    for i in range(0,np.int(np.round(n_pca/3) ) ):
        for j in range(0,3):
            ax_1[j,i].imshow(feature_im[counter].reshape([width*2+1,width*2+1,width*2+1])[:,:,2]) 
            ax_1[j,i].set_title('feature number: '+str(counter) )
            counter +=1
        
    plt.show()

    for i in range(6):

        fig_2 = plt.figure()
        ax_2 = plt.axes(projection='3d')
        ax_2.set_xlabel("x")
        ax_2.set_ylabel("y")
        ax_2.set_zlabel("z")
        ax_2.grid(True)
        im_1 = feature_im[i].reshape([width*2+1,width*2+1,width*2+1])
        ax_2.plot_surface(im_1[0],im_1[1],im_1[2], rstride=1, cstride=1,cmap='viridis', edgecolor='none')
        plt.show()

    my_dict = {}

    if phase =='fit':
        my_dict['pca_model'] = pca_model

    my_dict = { 'features': windows_transform.tolist(),
                'mask_idx': mask_idx.tolist(),
                'image_shape' : dti_shape            
    }

    with open(output_file+'/'+filename+'_bow'+'.pickle', 'wb') as f:
        pickle.dump(my_dict, f)

    if phase =='fit':
        return pca_model

    else:
        return

def mri_kmeans(input_file,out_file,filename,_model,k,phase):

    with open(input_file,'rb') as f:
        raw_data = pickle.load(f)

    feat_mri = raw_data['features']

    if phase == 'fit' :
        clusters = _model.fit_predict(feat_mri)

    if phase == 'predict' :
        clusters = _model.predict(feat_mri)
    
    label_count = count_labels(clusters)

    raw_data['label_count'] = label_count.tolist()
    raw_data['label_list'] = clusters.tolist()
    raw_data['k'] = k
    del raw_data['features']

    with open(out_file+'/'+filename+'_kmm_labels'+'.pickle', 'wb') as f:
        pickle.dump(raw_data, f)

    return _model

def mri_gmm(input_file,n,out_file,filename):

    with open(input_file,'rb') as f:
        raw_data = pickle.load(f)

    feat_mri = raw_data['features']  
    _model = GMM()

    labels = _model.fit_predict(feat_mri,n,)
    label_count = count_labels(labels)

    raw_data['label_count'] = label_count.tolist()
    raw_data['label_list'] = labels.tolist()
    del raw_data['features']

    with open(out_file+'/'+filename+'gmm_labels'+'.pickle', 'wb') as f:
        pickle.dump(raw_data, f)

    return


def reconstruct_from_labels(input_file,out_file,filename):

    with open(input_file ,'rb') as f:
        raw_data = pickle.load(f)

    _labels = np.array(raw_data['label_list']).flatten() 
    mask_idx = np.array(raw_data['mask_idx'] )
    img_shape = np.array(raw_data['image_shape'])

    reconstruct_img = np.zeros(img_shape)

    for i in range(0,mask_idx.shape[0]):
            w_x = mask_idx[i,0]
            w_y = mask_idx[i,1]
            w_z = mask_idx[i,2]

            reconstruct_img[w_x,w_y,w_z] = _labels[i]

    img_finnish= reconstruct_img

    #write pickle file
    raw_data['image'] = img_finnish.tolist() 
    del raw_data['label_list']

    with open(out_file+'/'+filename+'_reconstruct'+'.pickle', 'wb') as f:
        pickle.dump(raw_data, f)

    return 


def diff_mask_labels(input_file_1,input_file_2,out_file):

    with open(input_file_1 ,'rb') as f:
        raw_data_1 = pickle.load(f)

    with open(input_file_2 ,'rb') as f:
        raw_data_2 = pickle.load(f)

    mask_dict ={}
    mask_dict['mask_idx'] = raw_data_1['mask_idx']
    mask_dict['image_predicted'] = raw_data_1['image']
    mask_dict['image_fit'] = raw_data_2['image']
    mask_dict['k'] = raw_data_1['k']
    k = raw_data_1['k']
    total_diff = 0

    for i in range(k):
        
        mask_1 = np.array(raw_data_1['image']) == i
        mask_2 = np.array(raw_data_2['image']) == i

        mask_diff = mask_1 < mask_2

        mask_dict['mask_'+str(i)] = mask_diff.tolist()

        num_diff = np.sum(mask_diff)
        total_diff +=num_diff

        print('misclasified '+'label '+str(i)+':'+str(num_diff))   

    print('total_misclassified: '+str(total_diff) )

    with open(out_file+'/'+'label_mask'+'.pickle', 'wb') as f:  
        pickle.dump(mask_dict, f)

    return


def find_cc_label(idx_per_label):
    
    idx_per_label = np.array(idx_per_label)
    
    max_idx = np.argmax(idx_per_label[:,-1])

    max_label_info = idx_per_label[max_idx,:]

    return max_label_info


def mri_cc(input_file,out_file):

    with open(input_file ,'rb') as f:
        raw_data = pickle.load(f)

    labels = raw_data['k']

    list_label = np.zeros([labels,3],dtype=int)

    for _label in range(labels):
        label_img = np.array(raw_data['mask_'+str(_label)]) 
        cc  = cc3d.connected_components(label_img,zeroth_pass=False)

        cc_sizes = list()

        print('total voxels in label: '+str(np.sum(label_img) ) )

        with open(out_file+'/'+str(_label)+'.pickle' ,'wb') as f:
            pickle.dump(cc,f)

        for i in range(1,np.max(cc)+1) :
            mask_per_cc = cc == i
            sum_cc = np.sum(mask_per_cc)
            cc_sizes.append(sum_cc)

        #print('cc number: '+str(len(cc_sizes) ) )

        if cc_sizes == []:
            cc_max = 0
            cc_idx = 0

            print('max cc: '+'none')
        
        else:
            cc_sizes = np.array(cc_sizes)
            cc_max = np.max(cc_sizes)
            cc_idx =np.argmax(cc_sizes)

            print('max cc: '+str(cc_max))
        
        my_list = [_label,cc_idx,cc_max]
        raw_data['cc_img_'+str(_label)] = cc == cc_idx
        list_label[_label,:] = my_list

    max_label_info = find_cc_label(list_label)
    max_label_info[1] = max_label_info[1] +1

    raw_data['max_label_info'] = max_label_info

    print('biggest cc_img : '+str(max_label_info[-1] ) )

    for _label in range(labels):
        os.remove(out_file+'/'+str(_label)+'.pickle')
        del raw_data['mask_'+str(_label)]

    with open(out_file+'/'+'cc_img'+'.pickle', 'wb') as f:  
        pickle.dump(raw_data, f)

    return 


def show_3d_img(input_file,_stat_mask,p_stats):

    with open(input_file ,'rb') as f:
        raw_data = pickle.load(f)

    mask_idx =np.array(raw_data['mask_idx'] )
    max_label_info = raw_data['max_label_info']
    k = raw_data['k']

    print('cc voxels: '+str(max_label_info[-1] ) )

    w_x_min = np.min(mask_idx[:,0] )
    w_x_max = np.max(mask_idx[:,0] )
    w_y_min = np.min(mask_idx[:,1] )
    w_y_max = np.max(mask_idx[:,1] )
    w_z_min = np.min(mask_idx[:,2] )
    w_z_max = np.max(mask_idx[:,2] )

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True)
    ax.voxels(_stat_mask[w_x_min:w_x_max,w_y_min:w_y_max,w_z_min:w_z_max],facecolors='#1f77b430', edgecolors='grey', shade=False)
    [ ax.voxels(raw_data['cc_img_'+str(i)][w_x_min:w_x_max,w_y_min:w_y_max,w_z_min:w_z_max], edgecolors='blue', shade=False) for i in range(k) if np.sum(raw_data['cc_img_'+str(i)] ) < 200  ] 
    ax.voxels(p_stats[w_x_min:w_x_max,w_y_min:w_y_max,w_z_min:w_z_max],facecolors='#1f77b430', edgecolors='red', shade=False)
    plt.show()

    return

   

def show_images(input_file,p_stats):

    with open(input_file ,'rb') as f:
        raw_data = pickle.load(f)

    ht = np.array(raw_data['image_fit'] )
    pk = np.array(raw_data['image_predicted'] )
    mask_idx = np.array(raw_data['mask_idx'] )
    
    slices_x = range(int(np.min(mask_idx[:,0]) ), int(np.max(mask_idx[:,0]) ), int(np.round( (np.max(mask_idx[:,0])-np.min(mask_idx[:,0]) ) /5  ) ) )
    slices_y = range(int(np.min(mask_idx[:,1]) ), int(np.max(mask_idx[:,1]) ), int(np.round( (np.max(mask_idx[:,1])-np.min(mask_idx[:,1]) ) /5  ) ) )
    slices_z = range(int(np.min(mask_idx[:,2]) ), int(np.max(mask_idx[:,2]) ), int(np.round( (np.max(mask_idx[:,2])-np.min(mask_idx[:,2]) ) /5  ) ) )

    fig_1, axes_1 = plt.subplots(3,5)
    fig_2, axes_2 = plt.subplots(3,5)
    fig_3, axes_3 = plt.subplots(3,5)
    fig_1.suptitle('pk vs ht vs differnce in yz slices')
    fig_2.suptitle('pk vs ht vs differnce in xy slices')
    fig_3.suptitle('pk vs ht vs differnce in zx slices')

    for j  in range(0,9): 

        for i in range(0,5):
            
            if j == 0 :

                axes_1[j, i].imshow(diff_label(pk[slices_x[i],:,:] ) )

            elif  j == 1  :

                axes_1[j, i].imshow(diff_label(ht[slices_x[i],:,:] ) )

            elif j == 2 :
    
                axes_1[j, i].imshow(diff_label(p_stats[slices_x[i],:,:] ) )

            elif  j == 3  :

                axes_2[j-3, i].imshow(diff_label(pk[:,:,slices_z[i] ] ) )

            elif j == 4 :
    
                axes_2[j-3, i].imshow(diff_label(ht[:,:,slices_z[i] ] ) )

            elif  j == 5  :

                axes_2[j-3, i].imshow(diff_label(p_stats[:,:,slices_z[i] ] ) )

            elif  j == 6  :
    
                axes_3[j-6, i].imshow(diff_label(pk[:,slices_y[i],:] ) )

            elif j == 7 :
    
                axes_3[j-6, i].imshow(diff_label(ht[:,slices_y[i],:] ) )

            elif  j == 8  :

                axes_3[j-6, i].imshow(diff_label(p_stats[:,slices_y[i],:] ) )
    plt.show()


    return








if __name__ == "__main__":
    import model
    model.main()