MRI_module
a dipy-based MRI library - handling a dataset of multiple subjects nifty files . pre-processing to connectivity analysis, a whole pipeline in one module. 

I recomend reading more on the different methods on www.dipy.org

PIPELINE PER SUBJECT:
brain extraction 
calculate fa,md,rd,ad maps 
'affine'/'non_linear' registration of maps to template
csd orientation model (odf)
deterministric fiber extraction
fiber registration to template

STRUCTURE IMAGE:
structures 4D image registration to template image
 
 GETTING STARTED
 
 FILES
 you need to have these file to start the process:
 1.subdirectories with numbers in the title- the directories of each subject. their content:
 a. 4D image nifty( nii.gz) file - contains the raw image of the DTI scan
 b. b-values file( .bval) - contains the b-value of each scan
 c. b-vectors file (.bvec) - contains the b-vecors direction in each scan
 2. 3D template image nifty file (.nii.gz) - a standart space image used as a refference and used to registrate the subjects data in the process
 3. 4D structrues image nifty file (.nii.gz) - a 4D  binary array contains masks of structures of the brain that you wish to find connectivity features
    the first 3D image contains the image that you took the structures .  
 4. dipy_modeule_github.py - the module contains a class and some functions
 
 example:
 main directory - C\users\user\Deskrop\mri_project
 subfolders: 106, 168, 222, MNI_standart_space_template.nii.gz, structures.nii.gz, Dipy_module_github.py, my_code.py
 
 sub-directory of one subject - C\users\user\Deskrop\mri_project\106
 subfolders: 106_DTI.nii.gz , 106_bvals.bval, 106_bvec.bvec
 
 <img width="517" alt="folder_image" src="https://user-images.githubusercontent.com/66767504/124436003-39300c00-dd7e-11eb-9521-2c5cdee1cc1d.PNG">
<img width="518" alt="subfolders" src="https://user-images.githubusercontent.com/66767504/124436067-4a791880-dd7e-11eb-9bf3-d66a5992cb33.PNG">
