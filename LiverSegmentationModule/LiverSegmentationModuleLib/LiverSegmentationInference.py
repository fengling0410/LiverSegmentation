import monai
import numpy as np
import torch
import slicer
import os
import vtk

from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord, SpatialPadd)

def set_device(device):
    if device == "GPU" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("current device is:", device.type)
    return device


def initialize_and_load_model():
    model = UNETR(in_channels=1, 
        out_channels=2, 
        img_size=(96, 96, 96), 
        feature_size=16, 
        hidden_size=768, 
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.1,
        )
    model.load_state_dict(os.path.join(os.path.dirname(__file__), 'liver_model.pth'))
    print("UNETR model is initialized and loaded!")
    return model


def load_array_from_volume_node(volume_node):
    data = slicer.util.arrayFromVolume(volume_node)
    data = np.swapaxes(data, 0, 2)
    print("load data from input volume, the shape is: ", data.shape)

    m = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m)
    affine = slicer.util.arrayFromVTKMatrix(m)
    meta_data = {"affine": affine, "original_affine": affine, "spacial_shape": data.shape, 'original_spacing': volume_node.GetSpacing()}

    return {"image": data, "image_meta_dict": meta_data}



def set_inference_trans():
    val_trans = Compose([        
        AddChanneld(keys=["image"]),
        # orientation and down-sampling
        Spacingd(keys=["image"], pixdim=(2, 2, 2), mode=("bilinear")),
        Orientationd(keys=["image"], axcodes="RAS"),
        # SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
        ScaleIntensityRanged(keys=["image"],a_min=-160,a_max=240,b_min=0.0,b_max=1.0,clip=True),
        # CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"])
        ])
    return val_trans


def set_post_trans(original_spacing):
    post_trans = Compose([  
        AddChanneld(keys=["image"]), 
        Spacingd(keys=['image'], pixdim=original_spacing, mode="nearest")
    ])
    return post_trans


def inference(volumeNode, segmentationNode, device, model_path, overlap):
    with torch.no_grad():
        model_unetr = initialize_and_load_model()
        model_unetr.to(device)
        inputs = slicer.util.arrayFromVolume(volumeNode)
    
        # transform the inputs
        val_trans = set_inference_trans()
        inputs_trans = val_trans(inputs)
        model_input = inputs_trans['volume'].to(device)

        # start inference
        print("Run UNet model on input volume using sliding window inference")
        model_output = sliding_window_inference(model_input, (160, 160, 160), 4, model_unetr, overlap = overlap)
        if device.type == "cuda":
            model_output = model_output.detach().to("cpu").numpy()
        else:
            model_output = model_output.numpy()
            
        # get the discrete output
        model_pred = np.argmax(model_output, axis = 1)

        # post-transform the outputs
        original_spacing = inputs_trans["meta_data"]["original_spacing"]
        output_affine_matrix = inputs_trans["meta_data"]["affine"]
        output_trans = set_post_trans(original_spacing)(model_pred)
        label_map_output = output_trans["image"][0, :, :, :]
        print("label map shape :", label_map_output.shape)


        # update the segmentation
        # segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('Segment_1')
        # # segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
        # # check the dimension
        # if output_seg.shape == inputs.shape:
        #     return True
        # slicer.util.updateSegmentBinaryLabelmapFromArray(output_seg, segmentationNode, segmentId, volumeNode)

    
    return val_pred
    



