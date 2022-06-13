# 3D Slicer Liver Segmentation Extension
This is the repository for 3D slicer extension Liver Segmentation Module developed by Ling Feng at Dana Farber Cancer Institute. The inference model has a structure of UNETR [[1]](#1) and is trained using self-supervised learning. The model is pre-trained using image context restoration [[2]](#1) on 2500 [**TCIA**](https://www.cancerimagingarchive.net) abdomen CT scans, and the model is fine-tuned on [**LiTS**](https://competitions.codalab.org/competitions/17094) liver segmentation dataset which contains 130 annotated CT scan.

This extension is still under development and has not been distributed yet. To make use of the extension, please follow the instructions below. We welcome and appreciate any suggestions from the community.

### Download our Extension and Load it into 3D Slicer
You may download out extension through github directly. In order to load the extension into 3D Slicer and use it for liver segmentation task, you can go through **Module Finder** --> **Extension Wizard** --> **Select Extension** --> **choose the folder contains the extension** --> **Open**. After loading the module, you can use it by go through **Module** --> **Segmentation** --> **LiverSegmentation**. The user-interface will look like the following: ![alt text](https://github.com/fengling0410/LiverSegmentation/blob/main/Images/user_interface.png)

The parameters on the user-interface means:
- `Input Volume`: The CT Scan you want to segment.
- `Overlap`: Amount of overlap between scans. For cpu user, it is recommended to set the overlap to be 0.5, while gpu user could use a higher overlap such as 0.8.
- `Device`: User can choose between "gpu" or "cpu". If "gpu" is chosen but not available, the device will be set to "cpu" automatically.
- `Target Organ`: Current only "Liver" is supported. In the future, we envision our extension can be extended to incorporate multi-organ segmentation.
- `Output Segmentation`: The output SegmentationNode, which contains a segment named "Liver".

### Check and Load Dependencies
First, please make sure the required python packages are installed in your 3D Slicer environment. This extension requires the installation of **monai**, **torch**, **einops**, **nibabel**, and **scikit-image**. These packages can be installed manually using command `slicer.util.pip_install()` or by clicking `Check and Download Dependencies` button on the extension user-interface.

### Start Liver Segmentation
First, before starting doing segmentation, please make sure required python packages are installed by following the above instructions. Second, go to **Module** --> **Segmentations** and create a SegmentationNode and a sub-segment. You must name the sub-segment as "Liver". Please make sure the segment is created before clicking the `Apply` button. The segmentation structure should look like this: ![alt text](https://github.com/fengling0410/LiverSegmentation/blob/main/Images/segmentations.png)

For the input volume, user can load any abdomen/chest CT scan. User can also choose to play with our extension using our provided sample data `LiTS-Liver-Segmentation-0` under section **Download Sample Data** and label **Liver Segmentation**. The built-in sample data `CTChest` also works. After set the input volume node and output segmentation node, user could click the `Apply` button to start the inference, the process could take several minutes to 10 minutes depends on the size of the input CT scan. After the inference is done, user could freely edit the output segmentation by using the module `Segmentation Editor`. The output should look like the following:
![alt text](https://github.com/fengling0410/LiverSegmentation/blob/main/Images/output.png)

### Reference
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.

<a id="2">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.



