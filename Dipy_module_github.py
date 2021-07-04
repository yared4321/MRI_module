import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import stats as sc_stats

#### START LIBRARIES
from dipy.data import  (gradient_table,
                        get_fnames,
                        small_sphere,
                        default_sphere)
from dipy.io.gradients import read_bvals_bvecs
                               
from dipy.io.image import load_nifti , save_nifti
from dipy.segment.mask import median_otsu
from dipy.denoise.gibbs import gibbs_removal

#### csd and dti
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst,
                                   estimate_response,
                                   recursive_response )
import dipy.reconst.dti as dti
from dipy.tracking import utils
from dipy.reconst.shm import CsaOdfModel

### fiber tracking
from dipy.direction import (peaks_from_model,
                        DeterministicMaximumDirectionGetter,
                        ProbabilisticDirectionGetter)
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines, deform_streamlines, orient_by_rois
from dipy.tracking.life import transform_streamlines
from dipy.tracking.utils import transform_tracking_output
from dipy.tracking.streamline import apply_affine
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk 
from dipy.io.streamline import load_tractogram, save_tractogram

###transform
from dipy.align.reslice import reslice, affine_transform
from dipy.align import (affine_registration, center_of_mass, translation,
                        rigid, register_dwi_to_template)
from dipy.align.imaffine import (transform_centers_of_mass,
                                AffineMap,
                                MutualInformationMetric,
                                AffineRegistration,
                                transform_origins)                                
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric, SimilarityMetric
from dipy.align.streamlinear import transform_streamlines,center_streamlines
from dipy.align._public import transform_tracking_output

### visualisation
from dipy.viz import  regtools, window, actor, colormap, has_fury

def hardi_data(path):
    f_hardi_data, hardi_bval, hardi_bvec = get_fnames('stanford_hardi')
    data, affine = load_nifti(f_hardi_data)
    bvals, bvecs = read_bvals_bvecs(hardi_bval, hardi_bvec)
    
    bvals = list(bvals)
    str_bvals = ''
    
    for val in bvals:
        str_bvals += str(val)
        str_bvals +=' '
    
    bvecs = bvecs.T
    bvecs = list(bvecs)
    str_bvecs = ''
    
    for i in bvecs:
        for j in i:
            str_bvecs += str(j)
            str_bvecs += ' '

        str_bvecs += '\n'
    
    if not os.path.isdir(path):
        os.mkdir(path)

    save_nifti(path+'999\\999.nii.gz',data,affine)

    with open(path+'999\\'+'999.bval','w') as f:
        f.write(str_bvals)
    with open(path+'\\'+'999.bvec','w') as f:
        f.write(str_bvecs)

    return

def save_jason(data,path):
    
    with open(path, 'w') as f:
        json.dump(data, f)

    return

def load_jason(path):
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data


    with open(path, 'rb') as f:
        x = pickle.load(f)

    return np.array(x)

def box_im(im):

    im_shape = im.shape
    coords = np.array(np.where(im)).T

    w_x_min = np.min(coords[:,0] )
    w_x_max = np.max(coords[:,0] )
    w_y_min = np.min(coords[:,1] )
    w_y_max = np.max(coords[:,1] )
    w_z_min = np.min(coords[:,2] )
    w_z_max = np.max(coords[:,2] ) 

    new_im = im[w_x_min-1 :w_x_max+1,w_y_min-1 :w_y_max+1,w_z_min-1 :w_z_max+1].copy()

    return new_im

def make_isocenter(vox_size,origin_affine):

    target_isocenter = np.diag(np.array([-vox_size, vox_size, vox_size, 1]))
    origin_affine[0][3] = -origin_affine[0][3]
    origin_affine[1][3] = -origin_affine[1][3]
    origin_affine[2][3] = origin_affine[2][3]
    origin_affine[1][3] = origin_affine[1][3]

    return target_isocenter , origin_affine 

class MRI_module():

    def __init__(self,main_path,f_template,sub_dir = None,f_struct = None):
        
        self.fnames = [x for x in os.listdir(main_path) if os.path.isdir(x) and all(map(str.isdigit,x))]
        
        if not self.fnames :
            raise Exception('no subjects directories in main directory') 
            
        self.fnames = tuple(self.fnames)
        
        if sub_dir != None :
            self.sub_dir = sub_dir
        else: 
            self.sub_dir = None

        self.main_path = main_path
        self.stat_path = self.main_path + '\\stats'
        self.root = os.path.dirname(self.main_path)
        self.template = f_template
        self.struct = f_struct
        
        json_dict = [x for x in os.listdir(main_path) if x == 'data_dict'] 
        
        if json_dict != [] :
            self.data_dic = load_jason(self.main_path+'\\'+json_dict[-1])
        
        else:  
            self.data_dic = dict()
            self.data_dic['keys'] = list()
            self.data_dic['struct_reg'] = self.main_path+'\\'+'struct_reg.nii.gz'

            for name in self.fnames:
                num = int(name)
                self.data_dic['keys'].append(num)

                if self.sub_dir != None:
                    num_folder = self.main_path+ '\\'+ name+ '\\'+ self.sub_dir
                
                else:
                    num_folder = self.main_path+ '\\'+ name

                self.data_dic.update({str(num): [num_folder] })

                b_names = [x for x in os.listdir(num_folder) if x.endswith('.bval') or x.endswith('.bvec')  ] 
                bvec = [ x for x in b_names if x.endswith('.bvec')]
                bvec_f  = num_folder +'\\' + bvec[-1]
                bval =  [ name for name in b_names if name.endswith('.bval')]
                bval_f  = num_folder +'\\' + bval[-1]
                self.data_dic[str(num)].append([bval_f,bvec_f])
                
                data_name = [ data for data in os.listdir(num_folder) if data.endswith('.nii.gz') or data.endswith('.nii') ]
                data_path = num_folder+'\\'+data_name[0]
                
                path_endings = ['_bet','_b0_mask','_fa','_md','_rd','_ad','_odf','_peaks','_tractogram.trk']
                path_reg_endings = ['_fa_reg','_md_reg','_rd_reg','_ad_reg','_struct','_tractogram_reg.trk'] 
                file_type = '.nii.gz'
                
                paths = [ self.data_dic[str(num)][0]+'\\'+str(num)+path+file_type for path in path_endings if not path.endswith('.trk') ]
                paths.append(self.data_dic[str(num)][0]+'\\'+str(num)+path_endings[-1])
                paths.insert(0,data_path)
                
                reg_paths = [ self.data_dic[str(num)][0]+'\\'+str(num)+path+file_type for path in path_reg_endings if not path.endswith('.trk') ]
                reg_paths.append(self.data_dic[str(num)][0]+'\\'+str(num)+path_reg_endings[-1])
            
                self.data_dic[str(num)+'_paths'] = paths
                self.data_dic[str(num)+'_reg_paths'] = reg_paths
            
            save_jason(self.data_dic,self.main_path+'\\data_dict')

        return

    def add_path(self,num):

        json_dict = [x for x in os.listdir(self.main_path) if x == 'data_dict'] 
       
        if json_dict != [] :
            self.data_dic = load_jason(self.main_path+'\\'+json_dict[-1])
            if num in self.data_dic['keys']:
                raise Exception('there is a directory :'+ str(num)+' in the dictionary allready')
        else:
            raise Exception('no dictionary - you didnt initialize mri project') 

        num = int(num)
        self.data_dic['keys'].append(num)

        if self.sub_dir != None:
            num_folder = self.main_path+ '\\'+ num+ '\\'+ self.sub_dir
        else:
            num_folder = self.main_path+ '\\'+ str(num)

        self.data_dic.update({str(num): [num_folder] })

        b_names = [x for x in os.listdir(num_folder) if x.endswith('.bval') or x.endswith('.bvec')  ] 
        bvec = [ x for x in b_names if x.endswith('.bvec')]
        bvec_f  = num_folder +'\\' + bvec[-1]
        bval =  [ name for name in b_names if name.endswith('.bval')]
        bval_f  = num_folder +'\\' + bval[-1]
        self.data_dic[str(num)].append([bval_f,bvec_f])
        
        data_name = [ data for data in os.listdir(num_folder) if data.endswith('.nii.gz') or data.endswith('.nii') ]
        data_path = num_folder+'\\'+data_name[0]
        
        path_endings = ['_bet','_b0_mask','_fa','_md','_rd','_ad','_odf','_peaks','_tractogram.trk']
        path_reg_endings = ['_fa_reg','_md_reg','_rd_reg','_ad_reg','_struct','_tractogram_reg.trk'] 
        file_type = '.nii.gz'
            
        paths = [ self.data_dic[str(num)][0]+'\\'+str(num)+path+file_type for path in path_endings if not path.endswith('.trk') ]
        paths.append(self.data_dic[str(num)][0]+'\\'+str(num)+path_endings[-1])
        paths.insert(0,data_path)
        reg_paths = [ self.data_dic[str(num)][0]+'\\'+str(num)+path+file_type for path in path_reg_endings if not path.endswith('.trk') ]
        reg_paths.append(self.data_dic[str(num)][0]+'\\'+str(num)+path_reg_endings[-1])
    
        self.data_dic[str(num)+'_paths'] = paths
        self.data_dic[str(num)+'_reg_paths'] = reg_paths
            
        save_jason(self.data_dic,self.main_path+'\\data_dict')
        print('folder added: ' +str(num))
        return
    
    def bet(self,num, _median_radius = 4 , _numpass = 4):
        
        print('brain extraction: '+str(num) )
        
        data_path = self.data_dic[str(num)+'_paths'][0] 
        bet_path = self.data_dic[str(num)+'_paths'][1] 
        b0_mask_path = self.data_dic[str(num)+'_paths'][2]  

        data, affine = load_nifti(data_path)
        bet_mask, b0_mask = median_otsu(data,vol_idx=[1], median_radius= _median_radius, numpass= _numpass)

        save_nifti(bet_path,bet_mask.astype(np.float32),affine)
        save_nifti(b0_mask_path,b0_mask.astype(np.float32),affine)
        
        return
    
    def make_maps(self,num) :

        print('constract maps'+' : '+str(num))
        start_time = time.time()

        fa_path = self.data_dic[str(num)+'_paths'][3] 
        md_path = self.data_dic[str(num)+'_paths'][4] 
        rd_path = self.data_dic[str(num)+'_paths'][5]
        ad_path = self.data_dic[str(num)+'_paths'][6]
        bet_path = self.data_dic[str(num)+'_paths'][1] 

        bet_mask, affine = load_nifti(bet_path)
    
        bval_f = self.data_dic[str(num)][1][0]
        bvec_f = self.data_dic[str(num)][1][1] 

        bvals, bvecs = read_bvals_bvecs(bval_f, bvec_f)
        gtab = gradient_table(bvals, bvecs)
        
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(bet_mask)
        
        MD = tenfit.md
        FA = tenfit.fa
        RD = tenfit.rd
        AD = tenfit.ad

        print('saving maps files')
        save_nifti(fa_path,FA.astype(np.float32),affine)
        save_nifti(md_path,MD.astype(np.float32),affine)
        save_nifti(rd_path,RD.astype(np.float32),affine)
        save_nifti(ad_path,AD.astype(np.float32),affine)

        end_time = time.time()
        print('total computation time :'+ str(round(end_time-start_time,2)) +' sec')
   
    def make_csd(self,num,_sphere = 'default_sphere' ,_fa_thr = 0.7):
        
        start_time = time.time()

        f_data = self.data_dic[str(num)+'_paths'][1] 
        f_gfa = self.data_dic[str(num)+'_paths'][3] 
        odf_path = self.data_dic[str(num)+'_paths'][-3] 

        data, _ = load_nifti(f_data)
        fa_map, fa_affine = load_nifti(f_gfa)
        b0_mask = fa_map > 0

        print('constract model'+' : '+str(num))
        
        bval_f,bvec_f = self.data_dic[str(num)][1]
        bvals, bvecs = read_bvals_bvecs(bval_f, bvec_f)
        gtab = gradient_table(bvals, bvecs)

        response, _ = auto_response_ssst(gtab, data , roi_radii=10, fa_thr=_fa_thr)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6,convergence=5)

        print('fitiing odf ')
        
        csd_fit= csd_model.fit(data, mask=b0_mask)

        if _sphere == 'default_sphere' :
            csd_odf = csd_fit.odf(default_sphere)
        else:
            csd_odf = csd_fit.odf(_sphere)

        ### IMAGE ###
        # scene = window.Scene()

        # fodf_spheres = actor.odf_slicer(csd_odf, sphere= default_sphere, scale=0.7,
        #                     norm=True, colormap='plasma')

        # vol_actor = actor.slicer(fa_map)
        # vol_actor.display(z =15)
        # scene.add(vol_actor)
        # scene.add(fodf_spheres)
        
        # window.record(scene, size=(600, 600))
        # window.show(scene)

        ### DONE IMAGE ###
         
        print('saving files')
        save_nifti(odf_path,csd_odf.astype(np.float32),fa_affine)

        end_time = time.time()
        print('total processing time : ' + str(round(end_time-start_time,2))+' sec')

    def create_csd_model(self,num,_fa_thr=0.7):
        f_bet = self.data_dic[str(num)+'_paths'][1] 
        # f_fa = self.data_dic[str(num)+'_paths'][3] 
        bval_f,bvec_f = self.data_dic[num][1]
        bvals, bvecs = read_bvals_bvecs(bval_f, bvec_f)
        gtab = gradient_table(bvals, bvecs)

        # fa ,_= load_nifti(f_fa)
        # fa = fa > 0
        bet_mask, _ = load_nifti(f_bet)
        response, _ = auto_response_ssst(gtab, bet_mask , roi_radii=10,fa_thr = _fa_thr)
        _csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6,convergence=50)

        return _csd_model
    
    def make_peaks(self,num) :
        start_time = time.time()
        num = str(num)
        current_dir = self.data_dic[num][0]
        f_data = self.data_dic[str(num)+'_paths'][0] 
        f_bet = self.data_dic[str(num)+'_paths'][1] 
        f_b0_mask = self.data_dic[str(num)+'_paths'][2] 
        f_gfa = self.data_dic[str(num)+'_paths'][3] 
        odf_path = self.data_dic[str(num)+'_paths'][-3] 
        odf_peaks_path = self.data_dic[str(num)+'_paths'][-2] 
        streamlines_path = self.data_dic[str(num)+'_paths'][-1] 

        data, affine = load_nifti(f_data)
        csd_model = self.create_csd_model(num)
        csd_peaks = peaks_from_model(model=csd_model,
                    data=data,
                    sphere=default_sphere,
                    relative_peak_threshold=.5,
                    min_separation_angle=25,
                    parallel=True)


        print('saving files')
        save_nifti(odf_peaks_path,csd_peaks.peak_dirs.astype(np.float32),affine)
        
        end_time = time.time()
        print('total processing time : ' + str(round(end_time-start_time,2))+' sec')

        return

    def make_streamlines(self,num,fa_thr = 0.5, thr_stop = 0.25):
        
        start_time = time.time()
        print('start streamlines process: num '+str(num))

        f_b0_mask = self.data_dic[str(num)+'_paths'][2] 
        f_gfa = self.data_dic[str(num)+'_paths'][3] 
        odf_path = self.data_dic[str(num)+'_paths'][-3] 
        streamlines_path = self.data_dic[str(num)+'_paths'][-1] 

        b0_mask, _ = load_nifti(f_b0_mask)

        coords = np.array(np.where(b0_mask)).T
        min_z = np.min(coords[:,-1])
        max_z = np.max(coords[:,-1])
        min_x = np.min(coords[:,0])
        max_x = np.max(coords[:,0])
        min_y = np.min(coords[:,1])
        max_y = np.max(coords[:,1])

        b0_like = np.zeros_like(b0_mask)
        b0_like[min_x:max_x,min_y:max_y,min_z:max_z] = 1
        
        gfa, affine, img = load_nifti(f_gfa,return_img= True)
        csd_odf,_, _ = load_nifti(odf_path,return_img= True)
        
        seed_mask = gfa >= fa_thr*np.max(gfa)
        seed_mask = seed_mask*b0_like

        seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
        stopping_criterion = ThresholdStoppingCriterion(gfa, thr_stop)
        pmf = csd_odf.clip(min=0)
        det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                        sphere=default_sphere)
        print('start streamline generator')
        
        streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                            affine, step_size=0.5)
        streamlines = Streamlines(streamline_generator)
        sft = StatefulTractogram(streamlines, img, Space.RASMM)
        sft.remove_invalid_streamlines()

        print('saving streamline')
        
        save_tractogram(sft,streamlines_path)
        
        end_time = time.time()
        print('total processing time : ' + str(round(end_time-start_time,2))+' sec, num: '+str(num))

        return

    def reg_to_template(self,_type,reg,old_zooms =(1.,1.,1.), new_zooms = (1.,1.,1.),show = False,save= False):
        
        start_time = time.time()
        template,template_affine,template_img = load_nifti(self.template,return_img= True) 
        temp_shape = list(template.shape)
    
        
        if isinstance(_type,(np.int)) :
            num = _type

            print('start transform: '+str(num))
            
            f_files = [ self.data_dic[str(num)+'_paths'][i] for i in range(3,7) ] 
            f_reg = [ self.data_dic[str(num)+'_reg_paths'][i] for i in range(0,4) ] 

            data_list = [ load_nifti(f_files[i],return_img= True) for i in range(4) ] 
        
            if old_zooms[0] != new_zooms[0]:
                data , data_affine = reslice(np.asarray(data_list[0][0]),data_list[0][1],
                                        old_zooms, new_zooms) 
                new_shape = data.shape
                data_list = [ list(x) for x in data_list ]
                for i in range(len(data_list)):
                    data_list[i][0],_ = reslice(np.asarray(data_list[i][0]),data_affine,
                                        old_zooms, new_zooms) 

            else:
                data, data_affine, data_img = data_list[1][0] , data_list[1][1], data_list[1][2]

            nbins = 64
            c_of_mass = transform_centers_of_mass(template, template_affine,
                                        data, data_affine)
            self.data_dic[str(num)+'_com'] = c_of_mass
            
            print('c_o_m transformation')
            
            sampling_prop = None
            metric = MutualInformationMetric(nbins)
            level_iters = [10000, 1000, 100]
            sigmas = [3.0, 1.0, 0.0]
            factors = [4, 2, 1]   
            affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

            transform = TranslationTransform3D()
            params0 = None
            starting_affine = c_of_mass.affine
            
            print('affine transformation ')
            
            translation = affreg.optimize(template, data,transform, params0,
                                        template_affine, data_affine,
                                        starting_affine=starting_affine) 
            self.data_dic[str(num)+'_translation'] = translation
            
            if reg == 'affine':
                transformed = [ translation.transform(data_list[i][0]) for i in range(4) ]                 
                if show == True:
                    self.show_im('reg',num = None,data = transformed[0])
                
                if save == True :
                    [ save_nifti(f_reg[i],transformed[i],template_affine) for i in range(4) ]  
                
                end_time = time.time()
                print('total processing time : ' + str(round(end_time-start_time,2))+' sec')
                
            elif reg == 'non_linear':
                print('non-linear transformation ')
                
                metric = CCMetric(3)
                level_iters = [0,0,0]
                sdr = SymmetricDiffeomorphicRegistration(metric, level_iters,step_length= 2 , ss_sigma_factor= 4)
                mapping = sdr.optimize(template, data, template_affine, data_affine,
                                    translation.affine)
                self.data_dic[str(num)+'_mapping'] = mapping

                
                transformed = [ mapping.transform(data_list[i][0]) for i in range(4) ] 

                if show == True:
                    self.show_im('reg',num = None,data = transformed[0])
                
                if save == True :
                    [ save_nifti(self.data_dic[str(num)+'_reg_paths'][i],transformed[i],template_affine) for i in range(4) ]  

                end_time = time.time()
                print('total processing time : ' + str(round(end_time-start_time,2))+' sec')
            
            return
            
        if _type == 'struct':
            print('start transform: '+_type)
            
            f_struct = self.data_dic['struct_reg']
            self.data_dic['template_path'] = f_struct 
            _data, data_affine, data_img = load_nifti(self.struct,return_img= True)
            
            reg_struct = np.zeros((temp_shape[0], temp_shape[1],temp_shape[2],_data.shape[-1]) )

            first_struct, data_affine = reslice(np.asarray(_data[:,:,:,0]),data_affine,
                            old_zooms, new_zooms)
            new_shape = first_struct.shape 
            new_struct= np.zeros((_data.shape[-1],new_shape[0], new_shape[1],new_shape[2] ))             
            new_struct[0] = first_struct

            for i in range(1,_data.shape[-1]):
                new_struct[i], _ = reslice(np.asarray(_data[:,:,:,i]), template_affine,
                            old_zooms, new_zooms)

            c_of_mass = transform_centers_of_mass(template, template_affine,
                                        first_struct, data_affine)

            transformed = c_of_mass.transform(first_struct) 
            print('c_o_m transformation for: '+str(_type))
            
            nbins = 64
            sampling_prop = None
            metric = MutualInformationMetric(nbins, sampling_prop)
            level_iters = [10000, 1000, 100]
            sigmas = [3.0, 1.0, 0.0]
            factors = [4, 2, 1]   
            affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

            transform = TranslationTransform3D()
            params0 = None
            starting_affine = c_of_mass.affine
            print('affine transformation for: '+str(_type))
            translation = affreg.optimize(template, first_struct,transform, params0,
                                        template_affine, data_affine,
                                        starting_affine=starting_affine) 
            print('finnished affine transformed image: '+str(_type))

            if reg == 'affine':
                for i in range(_data.shape[-1]):
                    reg_struct[:,:,:,i] = translation.transform(new_struct[i])
                
                if show == True :
                    self.show_im('reg',num = None,data = reg_struct[:,:,:,0])
                if save == True :
                    save_nifti(f_struct,reg_struct,template_affine) 
                
                end_time = time.time()
                print('total processing time : ' + str(round(end_time-start_time,2))+' sec')
                
                return

            elif reg == 'non_linear':
                print('non-linear transformation : '+str(_type))
                metric = CCMetric(3)
                level_iters = [10, 10, 5]
                sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

                mapping = sdr.optimize(template, new_struct[0], template_affine, data_affine,
                                    translation.affine)

                for i in range(_data.shape[0]):
                    reg_struct[:,:,:,i] = mapping.transform(new_struct[i])
                
                if show == True :
                    self.show_im('reg',num = None,data = reg_struct[:,:,:,0])
                if save == True :
                    save_nifti(f_struct,reg_struct,template_affine)
                
                end_time = time.time()
                print('total processing time : ' + str(round(end_time-start_time,2))+' sec')
            
        return
    
    def reg_fibers(self,num,vox_diff = 1.) :
        
        start_time = time.time()
        self.reg_to_template(num,'non_linear')

        print('start fiber transform')

        template, template_affine, _ = load_nifti(self.template,return_img= True)
        # template = affine_transform(template,template_affine)
        
        fa, fa_affine, img = load_nifti(self.data_dic[str(num)+'_paths'][3],return_img= True)
        streamlines_path = self.data_dic[str(num)+'_paths'][-1]
        f_fibers = self.data_dic[str(num)+'_reg_paths'][-1]

        sft = load_tractogram(streamlines_path, img,to_space= Space.VOX)
        
        mapping = self.data_dic[str(num)+'_mapping']
        affine_map = self.data_dic[str(num)+'_translation']

        print('transform fibers: '+str(num))
       
        # affine_map = transform_origins(template, template_affine, fa, fa_affine)
        adjusted_affine = affine_map.affine.copy()
        adjusted_affine[0][3] = adjusted_affine[0][3] 
        adjusted_affine[1][3] = -adjusted_affine[1][3]/vox_diff
        adjusted_affine[2][3] = adjusted_affine[2][3]/ vox_diff**2 

        adjusted_field = mapping.get_forward_field()[-1:]

        target_isocenter, _ = make_isocenter(vox_diff,affine_map.affine.copy()) 
        
        mni_streamlines = deform_streamlines(
                                sft.streamlines, deform_field= adjusted_field,
                                stream_to_current_grid= target_isocenter,
                                current_grid_to_world= adjusted_affine, 
                                stream_to_ref_grid= target_isocenter,
                                ref_grid_to_world= np.eye(4)
                                )

        # mni_streamlines = transform_streamlines(sft.streamlines ,affine_map.affine )
        template_nifty  = nib.Nifti1Image(template, affine=template_affine)
        
        sft_2 = StatefulTractogram(mni_streamlines, template_nifty, Space.RASMM)
        sft_2.remove_invalid_streamlines()
        save_tractogram(sft_2,f_fibers , bbox_valid_check=False)

        print('fiber transformation complete: ')
        
        end_time = time.time()
        print('total processing time : ' + str(round(end_time-start_time,2))+' sec')
        self.show_im('streamline_reg',num)
        return

    def C_A(self,num,dim = 'world'):
        
        num = str(num)
        start = time.time()
        print('start connectivity analysis: '+str(num))
        
        current_dir = self.data_dic[num][0]
        C_A_dir = current_dir+'\\'+str(num)+'_C_A'
        f_struct = self.data_dic['struct_reg']

        print('preparing data')
        
        if not os.path.exists(C_A_dir):
            os.mkdir(C_A_dir)

        if dim =='world' : 
            f_data = self.data_dic[str(num)+'_paths'][3]
            f_struct = self.struct
            f_streamline = self.data_dic[str(num)+'_paths'][-1]
            chosen_space = Space.VOX
            
        elif dim == 'template':
            f_data = self.data_dic[str(num)+'_reg_paths'][0]
            f_struct = self.data_dic['struct_reg']
            f_streamline = self.data_dic[str(num)+'_reg_paths'][-1]
            chosen_space = Space.RASMM

        
        data ,data_affine,img = load_nifti(f_data,return_img= True)
        struct , struct_affine = load_nifti(f_struct)

        m_data = data > 0 
        streamlines = load_tractogram(f_streamline,img, to_space=chosen_space)
        labels = np.zeros_like(struct[:,:,:,0])
        num_areas = struct.shape[-1]
        mask_streamlines = streamlines.streamlines

        for i in range(1,num_areas):
            mask = m_data*struct[:,:,:,i]
            labels = np.where(mask,i,labels)

        labels = labels.astype(np.int)

        #labels
        mask_seeds = labels > 0 

        print('exclude fibers ')
        
        mask_streamlines = utils.target(streamlines.streamlines, struct_affine, mask_seeds)
        mask_streamlines = Streamlines(mask_streamlines)

        cc_ROI_actor = list()

        color = colormap.line_colors(mask_streamlines)
        cc_streamlines_actor = actor.line(mask_streamlines,
                                        colormap.line_colors(mask_streamlines))
        for i in range(0,np.max(labels) ):
            a = np.random.random(3)
            cc_ROI_actor.append(actor.contour_from_roi(labels == i, color=(a[0], a[1], a[2]),
                                            opacity=0.5) )

        vol_actor = actor.slicer(data)
        vol_actor.display(z=40)
        scene = window.Scene()
        scene.add(vol_actor)
        scene.add(cc_streamlines_actor)
        for i in range(0,np.max(labels) ):
            scene.add(cc_ROI_actor[i])
        # scene.add(cc_ROI_actor_1)
        window.show(scene)
        scene.set_camera(position=[-1, 0, 0], focal_point=[0, 0, 0], view_up=[0, 0, 1])

        print('build connectivity matrix')

        # # # 0 is backround (not included when saving track
        # # #  symetry make u loose some tracts - look at the tutorial in Dipy website)

        M, grouping = utils.connectivity_matrix(mask_streamlines, struct_affine,
                                        labels.astype(np.uint8),
                                        inclusive= True,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.log10(1+np.array(M)), interpolation='nearest')

        plt.title('connectivity matrix - log scale')
        plt.xticks(range(0,num_areas))
        plt.xlabel('ROI labels')
        plt.yticks(range(0,num_areas))
        plt.ylabel('ROI labels')
        plt.colorbar()
        plt.show()

        shape = data.shape
        dm_tot = np.zeros((shape[0],shape[1],shape[2],struct.shape[-1]-1))
        
        for i in range(num_areas-1) :
            for j in range(num_areas-1) :
                track = grouping[i+1, j+1]
                dm_tot[:,:,:,j] = utils.density_map(track, struct_affine, shape)
            save_nifti(C_A_dir+'\\'+str(num)+'_'+str(i)+'_track.nii', dm_tot.astype("int16"), data_affine)

        end = time.time()
        print('finnished connectivity analysis')
        print('total time: '+str(end-start)+' seconds')
        return     

    def prepare_stats(self,states,_mean = False):

        if type(states) != type(dict()) :
            raise Exception('dictionary needed {num:status} ')
            
        print('prepare stats folder for statistic analysis ')
        start_time = time.time()

        if not os.path.isdir(self.stat_path): 
                os.mkdir(self.stat_path)
        
        fa_reg_path = self.data_dic[str(self.data_dic['keys'][0])+'_reg_paths'][0] 
        fa_mask , affine_map = load_nifti(fa_reg_path)
        fa_shape = fa_mask.shape
        total_patients = len(self.data_dic['keys'])

        fa_maps_org = np.zeros((fa_shape[0],fa_shape[1],fa_shape[2],total_patients))
        md_maps_org = np.zeros_like(fa_maps_org)
        rd_maps_org = np.zeros_like(fa_maps_org)
        ad_maps_org = np.zeros_like(fa_maps_org)

        density_maps = np.zeros((fa_shape[0],fa_shape[1],fa_shape[2],8,8))
        density_maps = np.array(density_maps)
        
        c = 0 
        p = total_patients - 1

        for num in self.data_dic['keys']:
            
            num = str(num)
            state = states[num]
            fa_path = self.data_dic[str(num)+'_reg_paths'][0] 
            md_path = self.data_dic[str(num)+'_reg_paths'][1] 
            rd_path = self.data_dic[str(num)+'_reg_paths'][2]
            ad_path = self.data_dic[str(num)+'_reg_paths'][3]

            if state == 'control' :
                fa_maps_org[:,:,:,c], _ = load_nifti(fa_path)
                md_maps_org[:,:,:,c], _ = load_nifti(md_path)
                rd_maps_org[:,:,:,c], _ = load_nifti(rd_path)
                ad_maps_org[:,:,:,c], _ = load_nifti(ad_path)
                
                for j in range(8) :
                    density_path = self.data_dic[num][0]+'\\'+str(num)+'_C_A\\'+str(num)+'_'+str(j)+'_track.nii'
                    density_maps_num, _ = load_nifti(density_path)

                    for i in range(density_maps_num.shape[-1]) :
                        density_maps[:,:,:,j,i] += density_maps_num[:,:,:,i]
                
                c += 1

            elif state == 'patient':
                fa_maps_org[:,:,:,p], _ = load_nifti(fa_path)
                md_maps_org[:,:,:,p], _ = load_nifti(md_path)
                rd_maps_org[:,:,:,p], _ = load_nifti(rd_path)
                ad_maps_org[:,:,:,p], _ = load_nifti(ad_path)
                
                for j in range(8) :
                    density_path = self.data_dic[num][0]+'\\'+str(num)+'_C_A\\'+str(num)+'_'+str(j)+'_track.nii'
                    density_maps_num, _ = load_nifti(density_path)

                    for i in range(density_maps_num.shape[-1]) :
                        density_maps[:,:,:,j,i] += density_maps_num[:,:,:,i]

                p -= 1
        
        #saving images 
        print('saving images')
        save_nifti(self.stat_path+'//fa_all.nii', fa_maps_org , affine_map)
        save_nifti(self.stat_path+'//md_all.nii', md_maps_org.astype(np.float32) , affine_map)
        save_nifti(self.stat_path+'//rd_all.nii', rd_maps_org.astype(np.float32) , affine_map)
        save_nifti(self.stat_path+'//ad_all.nii', ad_maps_org.astype(np.float32) , affine_map)
        save_nifti(self.stat_path+'//density_all.nii', (density_maps/total_patients).astype(np.float32) , affine_map)
        
        if _mean == True:
            save_nifti(self.stat_path+'//mean_fa.nii', np.mean(fa_maps_org,axis = -1) , affine_map)
        
        end_time = time.time()
        print('total time: '+str(round(end_time-start_time,2))+' seconds')
        
        return (c-1,p+1)

    def t_tests(self,group_sum,test_list =['less','greater','greater','less'] ) :

        stat_path = self.stat_path
        fa_stat_path = self.stat_path+'//fa_all.nii'
        md_stat_path = self.stat_path+'//md_all.nii'
        rd_stat_path = self.stat_path+'//rd_all.nii'
        ad_stat_path = self.stat_path+'//ad_all.nii'
        density_stat_path = self.stat_path+'//density_all.nii'

        fa_data, _ = load_nifti(fa_stat_path)
        md_data, _ = load_nifti(md_stat_path)
        rd_data, _ = load_nifti(rd_stat_path)
        ad_data, _ = load_nifti(ad_stat_path)
        density_data, _ = load_nifti(density_stat_path)

        map_list = [fa_data,md_data,rd_data,ad_data]
        map_name = ['FA','MD','RD','AD']
            
        dens_shape = density_data.shape
        
        healthy_dict = {}
        patient_dict = {}

        num_test = 0
        for data_map in map_list :
            mean_mat = np.ones((dens_shape[-2],dens_shape[-1]))
            var_mat = np.ones((dens_shape[-2],dens_shape[-1]))

            healthy_dict['mean_'+map_name[num_test]] = list()
            healthy_dict['var_'+map_name[num_test]] = list()
            patient_dict['mean_'+map_name[num_test]] = list() 
            patient_dict['var_'+map_name[num_test]] = list()

            for j in range(dens_shape[-2]):
                for i in range(dens_shape[-1]):
                    mask_voxels = density_data[:,:,:,j,i] > 0
                    
                    if np.sum(mask_voxels) == 0 :
                        continue
                    
                    healthy_map = [ data_map[:,:,:,m] * mask_voxels for m in range(group_sum[0]) ] 
                    healthy_dict['mean_'+map_name[num_test]].append([ np.mean(healthy_map[m]) for m in range(len(healthy_map))] )
                    healthy_dict['var_'+map_name[num_test]].append([ np.var(healthy_map[m]) for m in range(len(healthy_map))] )
                    
                    patient_map = [ data_map[:,:,:,m] * mask_voxels for m in range(group_sum[0],group_sum[1]+1) ] 
                    patient_dict['mean_'+map_name[num_test]].append([ np.mean(patient_map[m]) for m in range(len(patient_map))] )
                    patient_dict['var_'+map_name[num_test]].append([ np.var(patient_map[m]) for m in range(len(patient_map))] )
                             
                    _, mean_mat[j,i] = sc_stats.ttest_ind(patient_dict['mean_'+map_name[num_test]][-1],healthy_dict['mean_'+map_name[num_test]][-1],equal_var=False, alternative=test_list[num_test])
                    _, var_mat[j,i] = sc_stats.ttest_ind(patient_dict['var_'+map_name[num_test]][-1],healthy_dict['var_'+map_name[num_test]][-1],equal_var=False)
                    # mean_mat[i,j] = mean_mat[j,i]
                    # var_mat[j,i] = var_mat[i,j]

            
            plt.figure(figsize= (10,10), )
            plt.imshow(mean_mat)
            plt.title(map_name[num_test]+' map Welch t-test p-value per ROI',fontsize = 19)
            plt.xticks(range(0,8))
            plt.xlabel('ROI labels',fontsize = 16)
            plt.yticks(range(0,8))
            plt.ylabel('ROI labels',fontsize = 16)
            plt.colorbar()
            plt.show()

            plt.imshow(var_mat)
            plt.title(map_name[num_test]+' var map p-value per ROI')
            plt.xticks(range(0,8))
            plt.xlabel('ROI labels')
            plt.yticks(range(0,8))
            plt.ylabel('ROI labels')
            plt.colorbar()
            plt.show()

            figure, axes = plt.subplots(1,2)
            axes[0].boxplot(healthy_dict['mean_'+map_name[num_test]])
            axes[0].set_title('healthy mean '+map_name[num_test])
            axes[1].boxplot(patient_dict['mean_'+map_name[num_test]])
            axes[1].set_title('parkinson mean '+map_name[num_test])
            plt.show()

            num_test += 1

        return

    def show_im(self,_type,num = None, data = None):
   
        if _type == 'streamline' or _type == 'fa' or _type == 'odf' and isinstance(num,np.int):
            current_dir = self.data_dic[str(num)][1]
            f_b0_mask = self.data_dic[str(num)+'_paths'][2] 
            f_gfa = self.data_dic[str(num)+'_paths'][3]
            
            fa, fa_affine,img = load_nifti(f_gfa,return_img= True)
            # fa = np.rot90(fa,2,(2,1))
            fa_idx = np.array(np.where(fa)).T
            w_x_min = np.min(fa_idx[:,0] )
            w_x_max = np.max(fa_idx[:,0] )
            w_y_min = np.min(fa_idx[:,1] )
            w_y_max = np.max(fa_idx[:,1] )
            w_z_min = np.min(fa_idx[:,2] )
            w_z_max = np.max(fa_idx[:,2] )

        if _type == 'fa':
            print(_type+': '+str(num))

            fig1, ax = plt.subplots(3, 3, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})
            fig1.subplots_adjust(hspace=0.3, wspace=0.05)
            
            [ ax[0,i].imshow(fa[w_x_min+i*int((w_x_max-w_x_min)/3),:,:]) for i in range(3)]
            [ ax[1,i].imshow(fa[:,w_y_min+i*int((w_y_max-w_y_min)/3),:]) for i in range(3)]
            [ ax[2,i].imshow(fa[:,:,w_z_min+i*int((w_z_max-w_z_min)/3)]) for i in range(3)]
            plt.show()

        elif _type =='odf':
            print(_type+': '+str(num))
            scene = window.Scene()

            odf_path = self.data_dic[str(num)+'_paths'][-3] 
            csd_odf,affine = load_nifti(odf_path)
            a = np.random.randint(20,50)

            fodf_spheres = actor.odf_slicer(csd_odf,affine,sphere= default_sphere, scale=0.7,
                                norm=True, colormap='plasma')
            vol_actor = actor.slicer(fa,fa_affine)
            vol_actor.display(z=a)
            scene.add(fodf_spheres)
            scene.add(vol_actor)
            window.record(scene, size=(600, 600))
            window.show(scene)

        elif _type == 'template':
            print(_type)
            temp, affine = load_nifti(self.template)
            
            temp_idx = np.array(np.where(temp)).T
            w_x_min = np.min(temp_idx[:,0] )
            w_x_max = np.max(temp_idx[:,0] )
            w_y_min = np.min(temp_idx[:,1] )
            w_y_max = np.max(temp_idx[:,1] )
            w_z_min = np.min(temp_idx[:,2] )
            w_z_max = np.max(temp_idx[:,2] )
            
            fig1, ax = plt.subplots(3, 3, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})
            fig1.subplots_adjust(hspace=0.3, wspace=0.05)
            
            [ ax[0,i].imshow(temp[w_x_min+i*int((w_x_max-w_x_min)/3),:,:]) for i in range(3)]
            [ ax[1,i].imshow(temp[:,w_y_min+i*int((w_y_max-w_y_min)/3),:]) for i in range(3)]
            [ ax[2,i].imshow(temp[:,:,w_z_min+i*int((w_z_max-w_z_min)/3)]) for i in range(3)]
            plt.show()
           
        elif _type == 'streamline':
            
            print(_type+': '+str(num))
            
            streamlines_path = self.data_dic[str(num)+'_paths'][-1]
            streamlines = load_tractogram(streamlines_path,img,to_space= Space.VOX)
            
            a = np.random.randint(w_z_min+30,w_z_max-20)
            if has_fury:
                scene = window.Scene()
                vol_actor = actor.slicer(fa)
                vol_actor.display(z=a)
                scene.add(actor.line(streamlines.streamlines, colormap.line_colors(streamlines.streamlines)))
                scene.add(vol_actor)
                window.record(scene, n_frames=1,size=(800, 800))
                window.show(scene)
        
        elif _type == 'streamline_reg':
            
            print(_type+': '+str(num))
            f_fibers = self.data_dic[str(num)+'_reg_paths'][-1]
            template, affine_template, img_temp = load_nifti(self.template,return_img= True)
            streamlines = load_tractogram(f_fibers,img_temp,to_space= Space.RASMM )
            
            if has_fury:
                scene = window.Scene()
                vol_actor = actor.slicer(template)
                vol_actor.display(x = 40)
                vol_actor2 = vol_actor.copy()
                vol_actor2.display(z=40)
                vol_actor3 = vol_actor2.copy()
                vol_actor2.display(y=40)
                scene.add(vol_actor)
                scene.add(vol_actor2)
                scene.add(vol_actor3)
                scene.add(actor.line(streamlines.streamlines, colormap.line_colors(streamlines.streamlines)))
                window.record(scene, n_frames=1,
                                size=(400, 400))
                window.show(scene)

        elif _type == 'struct' and isinstance(num,np.int):
            print(_type)
            struct, affine = load_nifti(self.struct)

            coords = np.array(np.where(struct[:,:,:,num])).T
            w_x_min = np.min(coords[:,0] )
            w_x_max = np.max(coords[:,0] )
            w_y_min = np.min(coords[:,1] )
            w_y_max = np.max(coords[:,1] )
            w_z_min = np.min(coords[:,2] )
            w_z_max = np.max(coords[:,2] )

            fig1, ax = plt.subplots(3, 3, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})
            fig1.subplots_adjust(hspace=0.3, wspace=0.05)
            
            [ ax[0,i].imshow(struct[:,:,:,num][w_x_min+i*int((w_x_max-w_x_min)/3),:,:]) for i in range(3)]
            [ ax[1,i].imshow(struct[:,:,:,num][:,w_y_min+i*int((w_y_max-w_y_min)/3),:]) for i in range(3)]
            [ ax[2,i].imshow(struct[:,:,:,num][:,:,w_z_min+i*int((w_z_max-w_z_min)/3)]) for i in range(3)]
            plt.show()

        elif _type == 'reg' and isinstance(data,(np.ndarray)):

            temp, affine = load_nifti(self.template)

            regtools.overlay_slices(temp, data, None, 0,
                                    "template", "Transformed")
            regtools.overlay_slices(temp, data, None, 1,
                                    "template", "Transformed")
            regtools.overlay_slices(temp, data, None, 2,
                                    "template", "Transformed")
            plt.show()
        return