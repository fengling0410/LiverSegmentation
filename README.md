# 3D Slicer Liver Segmentation Extension
This is the repository for 3D slicer extension Liver Segmentation Module developed by Ling Feng at Dana Farber Cancer Institute. This extension is still under development and has not been distributed yet. To make use of the extension, please follow the instructions below. We welcome and appreciate any suggestions from the community.

### Download our Extension and Load it into 3D Slicer
You may download out extension through github directly. In order to load the extension into 3D Slicer and use it for liver segmentation task, you can go through **Module Finder** --> **Extension Wizard** --> **Select Extension** --> **choose the folder contains the extension** --> **Open**. After loading the module, you can use it by go through **Module** --> **Segmentation** --> **LiverSegmentation**. The user-interface will look like the following:
![alt text]()


### Check and Load Dependencies
First, please make sure the required python packages are installed in your 3D Slicer environment. This extension requires the installation of **monai**, **torch**, **einops**, **nibabel**, and **scikit-image**. These packages can be installed manually using command `slicer.util.pip_install()` or by clicking `Check and Download Dependencies` button on the extension user-interface.

### Start Liver Segmentation
Before starting doing segmentation.
