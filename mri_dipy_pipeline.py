import Dipy_module_github as DP

#open folder with everything u need (listed in README file)  

main_path = '.'
f_template = '.\\FMRIB58_FA_STD_2mm.nii.gz'
f_struct = '.\\struct_reg.nii.gz'
f_sub_dir = 'DTI'
sub_dir = None

### hardi dti raw data available from the dipy library

# DP.hardi_data(main_path')
# a.add_path(999)

a = DP.MRI_module(main_path,f_template,sub_dir,f_struct)

### START HERE###
### PIPELINE ###
# PER SUBJECT:
# brain extraction 
# calculate fa,md,rd,ad maps 
# 'affine'/'non_linear' registration of maps to template
# csd orientation model (odf)
# deterministric fiber extraction
# fiber registration to template

# STRUCTURES:(x,y,z,number of structures)
# number = 0 is the whole breain u took the structures from

# structures regisration to template

# CODE
# [ a.bet(i,_median_radius = 4 , _numpass = 4) for i in a.data_dic['keys'] ]
# [ a.make_maps(i) for i in a.data_dic['keys'] ]
# [ a.reg_to_template(i,'affine',old_zooms =(1.,1.,1.), new_zooms = (1.,1.,1.),show = True,save= True) for i in a.data_dic['keys'] ]  
# [ a.make_csd(i,_sphere = 'default_sphere' ,_fa_thr = 0.7) for i in a.data_dic['keys'] ] 
# [ a.make_streamlines(i,fa_thr = 0.5, thr_stop = 0.25) for i in a.data_dic['keys'] ]  
# [ a.reg_fibers(i,vox_diff = 2.) for i in a.data_dic['keys'] ]
# a.reg_to_template('struct','affine',old_zooms =(1.,1.,1.), new_zooms = (1.,1.,1.),show = True,save =True)

### CONNECTIVITY ANALYSIS ###

# CODE
# [ a.C_A(i,dim ='template') for i in a.data_dic['keys'] ] 
# a.prepare_stats(states,_mean = True)

### generate dictionary of states {str(subject_number): status}
### for example : if u have 2 healthy subjects and 3 patients 
### then u need directory like this

# CODE
# states = dict()
# num_healthy = 2
# num_pat = 3
# [ states.update({str(a.data_dic['keys'][i]) :'control'}) for i in range(num_healthy) ]
# [ states.update({str(a.data_dic['keys'][i]) :'patient'})  for i in range(num_healthy,num_healthy+num_pat) ]

# a.t_tests(num_healthy+num_pat,test_list =['less','greater','greater','less'])

### just for 1 subject

# CODE
# a.bet(999)
# a.make_maps(999)
# a.reg_to_template(999,'affine',show= True,save = False)
# a.make_csd(999)
# a.make_streamlines(999)
# a.reg_fibers(999)
# a.C_A(999,dim ='template')

# SHOW IMAGES

# CODE
# a.show_im('odf',999)
# a.show_im('streamline', 999)
# a.show_im('streamline_reg',999)
# a.show_im('fa',999)
# a.show_im('struct',0)



