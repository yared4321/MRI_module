MRI_module
a dipy-based MRI library - handling a dataset of multiple subjects nifty files . pre-processing to connectivity analysis, a whole pipeline in one module. 

I recomend reading more on the different methods on www.dipy.org \n

PIPELINE PER SUBJECT: \n
brain extraction \n
calculate fa,md,rd,ad maps  \n
'affine'/'non_linear' registration of maps to template \n
csd orientation model (odf) \n
deterministric fiber extraction \n
fiber registration to template \n

STRUCTURE IMAGE: \n
structures 4D image registration to template image \n
 
 GETTING STARTED \n
 
 FILES \n
 you need to have these file to start the process: \n
 1.subdirectories with numbers in the title- the directories of each subject. their content: \n
 a. 4D image nifty( nii.gz) file - contains the raw image of the DTI scan \n
 b. b-values file( .bval) - contains the b-value of each scan \n
 c. b-vectors file (.bvec) - contains the b-vecors direction in each scan \n
 2. 3D template image nifty file (.nii.gz) - a standart space image used as a refference and used to registrate the subjects data in the process \n
 3. 4D structrues image nifty file (.nii.gz) - a 4D  binary array contains masks of structures of the brain that you wish to find connectivity features \n
    the first 3D image contains the image that you took the structures .   \n
 4. dipy_modeule_github.py - the module contains a class and some functions \n
 
 example: \n
 main directory - C\users\user\Deskrop\mri_project \n
 subfolders: 106, 168, 222, MNI_standart_space_template.nii.gz, structures.nii.gz, Dipy_module_github.py, my_code.py \n
 
 sub-directory of one subject - C\users\user\Deskrop\mri_project\106 \n
 subfolders: 106_DTI.nii.gz , 106_bvals.bval, 106_bvec.bvec \n
 
 <img width="517" alt="folder_image" src="https://user-images.githubusercontent.com/66767504/124436003-39300c00-dd7e-11eb-9521-2c5cdee1cc1d.PNG">
<img width="518" alt="subfolders" src="https://user-images.githubusercontent.com/66767504/124436067-4a791880-dd7e-11eb-9bf3-d66a5992cb33.PNG">
