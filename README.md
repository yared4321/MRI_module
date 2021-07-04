# MRI_module
a dipy-based MRI library - handling a dataset of multiple subjects nifty files . pre-processing to connectivity analysis, a whole pipeline in one module. 

a recomend reading more on the different methods on www.dipy.org

PIPELINE PER SUBJECT:
# brain extraction 
# calculate fa,md,rd,ad maps 
# 'affine'/'non_linear' registration of maps to template
# csd orientation model (odf)
# deterministric fiber extraction
# fiber registration to template

STRUCTURE IMAGE:
# structures 4D image registration to template image
