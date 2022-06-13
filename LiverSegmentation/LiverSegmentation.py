import logging
import os

import vtk
import ctk
import qt

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# LiverSegmentation
#

class LiverSegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "LiverSegmentation"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Segmentation"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Ling Feng (DFCI)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of a simple extension.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Ling Feng and Alexandra Chowdhury at Dana Farber Cancer Institute, and was partially funded by ......
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add LiTS sample to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # LiverSegmentationModule1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='Liver Segmentation',
    sampleName='LiTS-Liver-Segmentation-0',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'LiTS_Liver.png'),
    # Download URL and target file name
    uris="https://github.com/fengling0410/LiverSegmentation/releases/download/LiverSegmentation/LITS-Liver-Segmentation-0.nii",
    fileNames='LITS-Liver-Segmentation-0.nii',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:01008989e8bfb02292c1dc43944c979f653ab6132fa82371ba6b5bd2e22f97cf',
    # This node name will be used when the data set is loaded
    nodeNames='LiTS_Liver'
  )


#
# LiverSegmentationWidget
#

class LiverSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Setting up the Input section
    InputsCollapsibleButton = ctk.ctkCollapsibleButton()
    InputsCollapsibleButton.text = "Inputs"
    self.layout.addWidget(InputsCollapsibleButton)
    InputsLayout = qt.QFormLayout(InputsCollapsibleButton)

    # setting the InputVolumeSelector
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.inputSelector.addAttribute( "vtkMRMLScalarVolumeNode", "LabelMap", 0 )
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input CT scan." )
    InputsLayout.addRow("Input Volume: ", self.inputSelector)

    # Setting the overlap
    self.doubleSpinBox = qt.QDoubleSpinBox()
    self.doubleSpinBox.maximum = 1.00
    self.doubleSpinBox.minimum = 0.00
    self.doubleSpinBox.singleStep = 0.10
    self.doubleSpinBox.setToolTip("Choose the overlap for sliding-window inference.")
    InputsLayout.addRow("Overlap: ", self.doubleSpinBox)

    # Setting the Device
    self.deviceComboBox = qt.QComboBox()
    self.deviceComboBox.addItems(["CPU","GPU"])
    InputsLayout.addRow("Device: ", self.deviceComboBox)

    # Set the target organ, will be set to Liver now
    self.organComboBox = qt.QComboBox()
    self.organComboBox.addItems(["Liver"])
    InputsLayout.addRow("Target Organ: ", self.organComboBox)  

    # Setting up the Output section
    OutputsCollapsibleButton = ctk.ctkCollapsibleButton()
    OutputsCollapsibleButton.text = "Outputs"
    self.layout.addWidget(OutputsCollapsibleButton)
    OutputsLayout = qt.QFormLayout(OutputsCollapsibleButton) 

    # setting the OutputSegmentationSelector  
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ( ("vtkMRMLSegmentationNode"), "" )
    self.outputSelector.addEnabled = False
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.removeEnabled = False
    self.outputSelector.noneEnabled = False
    self.outputSelector.showHidden = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output segmentation." )
    OutputsLayout.addRow("Output Segmentation: ", self.outputSelector)

    # setting the dependency check
    self.dependencycheckButton = qt.QPushButton("Check and Download Dependencies")
    self.dependencycheckButton.toolTip = "Click to check and download required packages."
    self.dependencycheckButton.enabled = True
    OutputsLayout.addRow(self.dependencycheckButton)

    # setting the apply button
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the segmentation algorithm."
    self.applyButton.enabled = True
    OutputsLayout.addRow(self.applyButton)

    # Create logic
    self.logic = LiverSegmentationLogic()                

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI) # currentNodeChanged(vtkMRMLNode*) is the signal
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.deviceComboBox.connect("device(str)", self.updateParameterNodeFromGUI)
    self.doubleSpinBox.connect("overlap(double)", self.updateParameterNodeFromGUI)
    self.organComboBox.connect("organ(str)", self.updateParameterNodeFromGUI)

    # Buttons
    self.dependencycheckButton.connect('clicked(bool)', self.onDependencyButton)
    self.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()


  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()


  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()


  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)


  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)


  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()


  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())


  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()


  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.deviceComboBox.setCurrentText(self._parameterNode.GetParameter("Device"))
    self.organComboBox.setCurrentText(self._parameterNode.GetParameter("Organ"))
    self.doubleSpinBox.setValue(float(self._parameterNode.GetParameter("Overlap")))

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.applyButton.toolTip = "Run the segmentation algorithm."
      self.applyButton.enabled = True
    else:
      self.applyButton.toolTip = "Please select input and output volume nodes before clicking apply button."
      self.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False


  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
    self._parameterNode.SetNodeReferenceID("InputVolume", self.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.outputSelector.currentNodeID)
    self._parameterNode.SetParameter("Overlap", str(self.doubleSpinBox.value))
    self._parameterNode.SetParameter("Device", self.deviceComboBox.currentText)
    self._parameterNode.SetParameter("Organ", self.organComboBox.currentText)
    self._parameterNode.EndModify(wasModified)


  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
    # comment this for better error output
      self.logic.process(self.inputSelector.currentNode(), self.outputSelector.currentNode(), self.doubleSpinBox.value, self.deviceComboBox.currentText, self.organComboBox.currentText)


  def check_dependency(self):
    try:
      import torch
      import monai
      import numpy as np
      import einops
      import nibabel
      import skimage

      from monai.metrics import DiceMetric
      from monai.networks.nets import UNETR
      from monai.inferers import sliding_window_inference
      from monai.transforms import (AddChanneld, Compose, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord, EnsureTyped, Invertd, CropForegroundd)
      from monai.transforms.post.array import KeepLargestConnectedComponent
      return True
    except ImportError:
      return False
  
  def onDependencyButton(self):
    if self.check_dependency():
      print("All dependency passed!")
    else:
      print("Start loading required dependencies")
      progressDialog = slicer.util.createProgressDialog(maximum=0)
      for dep in ["monai", "torch", "einops", "nibabel","scikit-image"]: #"einops", "itk","nibabel"
        progressDialog.labelText = "Installing " + dep
        slicer.util.pip_install(dep)
      check_again = self.check_dependency()
      if check_again:
        progressDialog.labelText = "All dependencies are installed! " 
      else:
        progressDialog.labelText = "Still need extra packages! " 

# ---------------------------------------------------------------------------------------------------------------------------------------

#
# LiverSegmentationLogic
#

class LiverSegmentationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Overlap"):
      parameterNode.SetParameter("Overlap", "0.00")

  def process(self, inputVolume, outputVolume, overlap, device, organ):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """
    import torch
    import monai
    import numpy as np
    from monai.inferers import sliding_window_inference
    from monai.transforms.post.array import KeepLargestConnectedComponent

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    print("current device: ", device, type(device))
    print("overlap: ", overlap, type(overlap))
    print("organ: ", organ, type(organ))

    input_ok = self.check_input_volume(inputVolume)
    print("Input good? ", input_ok)
    output_ok = self.check_output_seg(inputVolume, outputVolume)
    print("Output good? ", output_ok)  

    # set up device
    c_device = self.set_device(device) # it returns a string
    print("After setting device, current device is: ", c_device.type)

    # initialize model
    inference_model = self.initialize_and_load_model(c_device)
    print("UNETR model is initialized!")

    # get input data
    input_dic = self.load_data_from_volume_node(inputVolume)
    # check the shape and affine
    print("load data from input volume, the shape is: ", input_dic["image"].shape)
    print("Type of input data: ", type(input_dic["image"])) # it is a nd array
    print("The affine is: ", input_dic["image_meta_dict"]["affine"])
    print("The maximum pixel intensity: ", np.max(input_dic["image"]))
    print("The minimum pixel intensity: ", np.min(input_dic["image"]))

    # get transform for input
    input_trans = self.get_input_trans()
    print("Input transformation created!")

    # get transform for output
    post_trans = self.get_output_trans(input_trans)
    print("Output transformation created!")

    # start inference
    model_inputs = input_trans(input_dic)
    print("Run UNet model on input volume using sliding window inference")
    print("input shape: ", model_inputs["image"].shape)
    with torch.no_grad():
      model_inputs["pred"] = sliding_window_inference(model_inputs["image"].to(c_device), (96, 96, 96), 4, inference_model, overlap = overlap)[0,:,:,:]
      model_outputs = post_trans(model_inputs)
      output = model_outputs["pred"]
      print("The output shape is: ", output.shape)
      print("inference finished!")

      if c_device.type == "cuda":
        output = output.detach().to("cpu").numpy()
      else:
        output = output.numpy()

    output = np.expand_dims(output, axis=0)
    print("check shape: ", output.shape)
    pred = np.argmax(output, axis = 1)
    pred = KeepLargestConnectedComponent(applied_labels=[1])(pred)[0,:,:,:]
    print("shape of the model pred: ", pred.shape)
    pred = pred.astype(np.int32)

    # export the segment array to the SegmentationNode
    segmentId = outputVolume.GetSegmentation().GetSegmentIdBySegmentName('Liver')    
    slicer.util.updateSegmentBinaryLabelmapFromArray(np.swapaxes(pred, 0, 2), outputVolume, segmentId, inputVolume)
    print("inference done!")

    # clean up memory
    for e in ["model_inputs", "model_outputs", "input_trans", "post_trans", "output", "pred", "input_dic"]:
      if e in locals():
        del locals()[e]
    print("clean memory done!")    

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

# -------------------------------------------------------------------------------------------------------------self defined functions in logic
  # check the volume input
  def check_input_volume(self, volumeNode):
        if not volumeNode:
          print("Input volume does not exist")
          return False
        if volumeNode.GetImageData() == None:
          print("cannot find image data")
          return False
        return True

  # check the segmentation output
  def check_output_seg(self, volumeNode, segmentationNode):
        if not segmentationNode:
          print("can't find the segmentation node")
          return False
        if segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('Liver') == None:
          print("Can't find the segment label named 'Liver")
          return False

        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('Liver')
        segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
        voxelArray = slicer.util.arrayFromVolume(volumeNode)
        if segmentArray.shape != voxelArray.shape:
          print("The shape of input volume does not match the shape of output segmentation label map")
          return False

        del voxelArray
        del segmentArray
        return True

  # set device for model inference
  def set_device(self, device):
    import torch
    if device == "GPU" and torch.cuda.is_available():
      current_device = torch.device("cuda")
    else:
      current_device = torch.device("cpu")
    return current_device

  # initialize UNETR model and load the pre-trained weights
  def initialize_and_load_model(self, current_device):
    from monai.networks.nets import UNETR
    # import einops
    import torch
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
        ).to(current_device)
    path_to_file = os.path.join(os.path.dirname(__file__), 'liver_model.pth')
    if os.path.exists(path_to_file):
      print("pre-trained weights found!")
      model.load_state_dict(torch.load(path_to_file, map_location=current_device))
    else:
      print("cannot find the pre-trained weights")
    return model

  # load the np.array and meta data
  def load_data_from_volume_node(self, volume_node):
    import numpy as np
    data = slicer.util.arrayFromVolume(volume_node)
    data = data.copy()
    data = np.swapaxes(data, 0, 2)

    m = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m)
    affine = slicer.util.arrayFromVTKMatrix(m)
    meta_data = {"affine": affine, "original_affine": affine, "spacial_shape": data.shape, 'original_spacing': volume_node.GetSpacing()}
    
    return {"image": data, "image_meta_dict": meta_data}

  # function to get the input transform
  def get_input_trans(self):
    from monai.transforms import (AddChanneld, Compose, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord, EnsureTyped, Invertd, CropForegroundd)
    val_trans = Compose([        
    AddChanneld(keys=["image"]),
    # orientation and down-sampling
    Spacingd(keys=["image"], pixdim=(2, 2, 2), mode=("bilinear")),
    Orientationd(keys=["image"], axcodes="RAS"),

    ScaleIntensityRanged(keys=["image"],a_min=-160,a_max=240,b_min=0.0,b_max=1.0,clip=True),
    CropForegroundd(keys=["image"], source_key="image"),
    AddChanneld(keys=["image"]),
    ToTensord(keys=["image"])
    ])
    return val_trans    

  # function to get the output transform
  def get_output_trans(self, input_trans):
    from monai.transforms import (AddChanneld, Compose, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord, EnsureTyped, Invertd, CropForegroundd)
    post_trans = Compose([EnsureTyped(keys="pred"),
                          Invertd(keys="pred",
                          transform=input_trans,
                          orig_keys="image",
                          meta_keys="pred_meta_dict",
                          orig_meta_keys="image_meta_dict",
                          meta_key_postfix="meta_dict",
                          nearest_interp=False,
                          to_tensor=True)
                          ])
    return post_trans

# ----------------------------------------------------------------------------------------------------------------

#
# LiverSegmentationTest
#

class LiverSegmentationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """

    self.setUp()
    self.test_LiverSegmentation1()

  def test_LiverSegmentation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('LiTS-Liver-Segmentation-0')
    self.delayDisplay('Loaded test data set')


    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    print("maximum pixel intensity: ", inputScalarRange[0])
    print("minimum pixel intensity: ", inputScalarRange[1])

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    Segment = slicer.vtkSegment()
    Segment.SetName("Liver")
    Segment.SetColor(0.3,0.3,0.5)

    outputVolume.GetSegmentation().AddSegment(Segment)
    self.delayDisplay('Output volume created')

    overlap = 0.5
    device = "CPU"
    organ = "Liver"

    # Test the module logic
    logic = LiverSegmentationLogic()

    # Test the algorithm 
    self.delayDisplay('Starting inference')
    logic.process(inputVolume, outputVolume, overlap, device, organ)

    self.delayDisplay('Test passed')
