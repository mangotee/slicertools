import os
import sys
import re
import fnmatch
import glob
import time
import inspect
import logging
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
import SimpleITK as sitk
import sitkUtils as sitku
from functools import partial

from os.path import join as osj
from os.path import normpath as osn

#from pip._internal import main as pipmain
#pipmain(['install', 'scipy'])

pnSlicerutils= '/Users/mangotee/Dropbox/Projects/SliceScriptedModules/dsgz_val_cissReg'
def slicerutilsImportCommand(pn=pnSlicerutils):
    print('import sys')
    print('sys.path.append("%s")'%(pn))
    print('from slicertools import slicerutil as su')

def pipInstallPrerequisites():
    import pip
    import subprocess
    import sys
    if 'pandas' not in sys.modules:
        #pip.main(['install', 'pandas'])
        p = subprocess.call([sys.executable, '-m', 'pip', 'install', 'pandas'])
        (stdoutdata, stderrdata) = p.communicate()
        print(stdoutdata)

'''
File search tools
'''
def locateFiles(pattern, root_path, level=0, tailOnly=False, sort=True):
    fl = np.array([])
    #for root, dirs, files in os.walk(root_path):
    for root, dirs, files in walklevel(root_path, level):
        for filename in fnmatch.filter(files, pattern):
            #print( os.path.join(root, filename))
            if tailOnly:
                fl = np.append(fl,filename)
            else:
                fl = np.append(fl,osnj(root, filename))
    if sort:
        fl = natural_sort(fl)
    return fl

# depends on pandas
# depends on pandas
def locateFilesDf(pattern, root_path, level=0, tailOnly=False, sorted=True):
    import pandas as pd
    ffl = locateFiles(pattern, root_path, level=level, tailOnly=tailOnly)
    #for root, dirs, files in os.walk(root_path):
    # create a pandas data frame with columns ff, root_path, subpath, filename
    df = pd.DataFrame(data=ffl,columns=['ff'])
    listSubpath = []
    listFn = []
    listFnRoot = []
    listFnExt = []
    for ff in ffl:
        sp,fn = os.path.split(ff.replace(root_path,''))
        # clean sp (subpath) from remaining path literals ('/','\\'), especially necessary in Windows
        sp = sp.replace('/','')
        sp = sp.replace('\\','')
        # store to list for df
        listSubpath.append(sp)
        listFn.append(fn)
        root,ext = splitext(fn)
        listFnRoot.append(root)
        listFnExt.append(ext)
    df = df.assign( subpath=listSubpath )
    df = df.assign( fn=listFn )
    df = df.assign( fn_root=listFnRoot )
    df = df.assign( fn_ext=listFnExt )
    df['pn'] = [os.path.split(x)[0] for x in df.ff]
    df['root_path'] = [root_path for x in df.ff]
    if sorted:
        df = df.sort_values(['ff'])
        df = df.reset_index(drop=True)
    return df
# regex tutorial:
# http://regextutorials.com/intro.html

def locateDirs(pattern, root_path, level=0):
    if not os.path.exists(root_path):
        raise ValueError('Directory ("%s") does not exist!'%root_path)
    pl = np.array([])
    for root, dirs, files in walklevel(root_path, level): #os.walk(root_path):
        for pathname in fnmatch.filter(dirs, pattern):
            #print( os.path.join(root, pathname))
            pl = np.append(pl,os.path.join(root, pathname))
    return pl


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def osnj(*args):
    return osn(osj(*args)).replace('\\','/')
    
def splitext(fn):
    if fn.endswith('.nii.gz'):
        root = fn.replace('.nii.gz','')
        ext = '.nii.gz'
        return (root, ext)
    else:
        return os.path.splitext(fn)
    
            
'''
TRANSFORMS
'''
def dependTransformAonB_preserve(childA, parentB):
    transformmatrix = vtk.vtkMatrix4x4()
    slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(childA, parentB, transformmatrix)
    parent_id = __GetID__(parentB)
    childA.SetAndObserveTransformNodeID(parent_id)
    childA.SetMatrixTransformToParent(transformmatrix)

def hardenTransformOfNode(nodeName):
    node = slicer.util.getNode(nodeName)
    if node is not None:
        logic = slicer.vtkSlicerTransformLogic()
        logic.hardenTransform(node)
    
def replaceTransform(tf1, tf2):
    tf1_parent = tf1.GetParentTransformNode()
    tf2_parent = tf2.GetParentTransformNode()

    tf1_children = getTransformableChildrenOfTransformableNode(tf1)
    tf2_children = getTransformableChildrenOfTransformableNode(tf2)

    # switch observed transform
    tf1.SetAndObserveTransformNodeID(__GetID__(tf2_parent))
    tf2.SetAndObserveTransformNodeID(__GetID__(tf1_parent))

    # switch children
    for tf1_c in tf1_children:
        tf1_c.SetAndObserveTransformNodeID(__GetID__(tf2))

    for tf2_c in tf2_children:
        tf2_c.SetAndObserveTransformNodeID(__GetID__(tf1))


def getTransformableChildrenOfTransformableNode(node):
    children = []
    for n in slicer.mrmlScene.GetNodesByClass("vtkMRMLTransformableNode"):
        if n is not None:
            parentTrf = n.GetParentTransformNode()
            if parentTrf == node:
                children.append(n)
        elif node is None:
            children.append(n)

    return children

def calculateInbetweenLinearTransform(nTrfSource,nTrfTarget,nTrfSourceToTarget):
    # Get concatenated transforms from source to target node. Source and
    # target nodes are allowed to be NULL, which means that transform
    # is the world transform.
    matrixAToB = vtk.vtkMatrix4x4()
    if nTrfSource is not None:
        a = nTrfSource.GetParentTransformNode()
    else:
        a = None
    if nTrfTarget is not None:
        b = nTrfTarget.GetParentTransformNode()
    else:
        b = None
    # calculate the matrix transform
    slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(a,b,matrixAToB)
    # set the matrix transform into target
    nTrfSourceToTarget.SetMatrixTransformToParent(matrixAToB)
    # documentation...
    # Andras Lasso:
    # (src: https://discourse.slicer.org/t/transformation-graph-for-quick-calculation-of-concatenated-transforms/326/5)
    # If you want to use transformAToB in a transform node using SetTransformToParent then
    # you have to deep-copy it using slicer.vtkMRMLTransformNode.DeepCopyTransform. If you
    # only have linear transforms it is much simpler and faster to use
    # slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes instead
    # slicer.vtkMRMLTransformNode.GetTransformBetweenNodes

def cloneVolumeNode(nameSource, nameCloned):
    ''' 
    Do not use the function "cloneVolumeNode". Use cloneNodeInSubjectHierarchy(nameSource, nameCloned) instead.
    '''
    print()
    #nSource = getOrCreateTransformNode(nameSource)
    #nCloned = getOrCreateTransformNode(nameCloned)
    #if nSource is not None:
    #    nCloned.Copy(nSource)
    #    nCloned.SetName(nameCloned)
    print('Do not use the function "cloneVolumeNode". Use cloneNodeInSubjectHierarchy(nameSource, nameCloned) instead.')
        
def cloneNodeInSubjectHierarchy(nameSource, nameCloned):
    # source: https://www.slicer.org/wiki/Documentation/Nightly/ScriptRepository#Clone_a_node
    nodeToClone = slicer.util.getNode(nameSource)
    # Clone the node
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    itemIDToClone = shNode.GetItemByDataNode(nodeToClone)
    clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
    clonedNode = shNode.GetItemDataNode(clonedItemID)
    clonedNode.SetName(nameCloned)
    return clonedNode

def cloneTransformNode(nameSource, nameCloned):
    nSource = getOrCreateTransformNode(nameSource)
    nCloned = getOrCreateTransformNode(nameCloned)
    #if nSource is not None:
    #    nCloned.Copy(nSource)
    #    nCloned.SetName(nameCloned)
    # transform of source
    m4x4Source = vtk.vtkMatrix4x4()
    nSource.GetMatrixTransformToParent(m4x4Source)
    # copy from target / cloned
    m4x4Cloned = vtk.vtkMatrix4x4()
    m4x4Cloned.DeepCopy(m4x4Source)
    # paste into cloned transform
    nCloned.SetMatrixTransformToParent(m4x4Cloned)
    nCloned.Modified()

def invertTransformNode(nameSource, nameInverted=None, suffix='_inv'):
    # inverts transform nameSource into nameInverted, clones nameSource transform if
    # nameInverted!=nameSource
    nSource = getOrCreateTransformNode(nameSource)
    if nSource is None:
        print('Input transform is "None" (%s), cannot be inverted!' % (nameSource))
        #raise ValueError('Input transform is "None" (%s), cannot be inverted!' % (nameSource))
    if nameInverted is None and suffix is None:
        print('nameInverted and suffix cannot both be None.')
        #raise ValueError('nameInverted and suffix cannot both be None. Please specify either nameInverted fully, or a suffix (will be appended to nameSource')
    if nameInverted is None:
        nameInverted = nameSource+suffix
    cloneTransformNode(nameSource, nameInverted) # creates a transform node if it doesn't exist yet
    nInverted = getOrCreateTransformNode(nameInverted)
    nInverted.Inverse()

def transformSetElement(nameTrf,row,col,val):
    nTrf = getNode(nameTrf)
    M = nTrf.GetMatrixTransformToParent()
    M.SetElement(row,col,val)
    nTrf.SetMatrixTransformToParent(M)
    
def transformSetZeroTranslation(nameTrf):
    transformSetElement(nameTrf,0,3,0.0)
    transformSetElement(nameTrf,1,3,0.0)
    transformSetElement(nameTrf,2,3,0.0)

def transformFrom4x4Matrix(nameTrf,M):
    nTrf = getOrCreateTransformNode(nameTrf)
    for row in range(4):
        for col in range(4):
            transformSetElement(nameTrf,row,col,M[row,col])
    return nTrf
    
def transformFromVtk4x4Matrix(nameTrf,trfVtk):
    nTrf = getOrCreateTransformNode(nameTrf)
    nTrf.SetMatrixTransformToParent(trfVtk)
    return nTrf
    
def transformFromVolume(nameVol,nameTrf):
    #nTrf = getOrCreateTransformNode(nameTrf)
    nVol = getOrCreateVolumeNode(nameVol)
    # set rotation
    trfVtk = vtk.vtkMatrix4x4()
    nVol.GetIJKToRASDirectionMatrix(trfVtk)
    # set translation
    O = nVol.GetOrigin()
    for i in range(3):
        trfVtk.SetElement(i, 3, O[i])
    # place in MRML tree
    nTrf = transformFromVtk4x4Matrix(nameTrf,trfVtk)
    return nTrf

def centerACPCTransform(nameTrfAcpc,nameFidlistAcpc):
    nTrfAcpc = getNode(nameTrfAcpc)
    nFidlistAcpc = getNode(nameFidlistAcpc)
    if nTrfAcpc is None:
        disp('function centerACPCTransform: ACPC transform node with name "%s" does not exist. Exiting...'%nameTrfAcpc)
    elif nFidlistAcpc is None:
        disp('function centerACPCTransform: ACPC transform node with name "%s" does not exist. Exiting...'%nameTrfAcpc)
    else:
        posAC=[0.0,0.0,0.0,0.0]
        posPC=[0.0,0.0,0.0,0.0]
        nFidlistAcpc.GetNthFiducialWorldCoordinates(0,posAC)
        nFidlistAcpc.GetNthFiducialWorldCoordinates(1,posPC)
        posAC = np.array(posAC)
        posPC = np.array(posPC)
        posMean = np.mean(np.stack((posAC,posPC),axis=1),axis=1)
        transformSetElement(nTrfAcpc.GetName(),0,3,-1.0*posMean[0])
        transformSetElement(nTrfAcpc.GetName(),1,3,-1.0*posMean[1])
        transformSetElement(nTrfAcpc.GetName(),2,3,-1.0*posMean[2])
        nTrfAcpc.Modified()

def transformNodeToITKTransform_Linear(nameTrf):
    nTrf = getOrCreateTransformNode(nameTrf)
    vtkTrf = nTrf.GetMatrixTransformToParent()
    itkTrf = sitk.AffineTransform(3) # transform in 3D
    # set translation
    T = []
    for i in range(3):
        T.append(vtkTrf.GetElement(i,3))
    itkTrf.SetTranslation(T)
    # set rotation
    R = []
    for i in range(3):
        for j in range(3):
            R.append(vtkTrf.GetElement(i,j))
    itkTrf.SetMatrix(R)
    return itkTrf
    

'''
Node Retrieval Stuff
'''


def __GetID__(node):
    if node is None:
        return None
    return node.GetID()

def getNode(name):
    # since Slicer 4.9.0, slicer.util.getNode() throws an error (!) when
    # there is no node of "name" in the MRML tree
    # this is a wrapper that catches the error and returns None
    # this restores the original behaviour
    if isinstance(name,str):
        try:
            node = slicer.util.getNode(name)
        except slicer.util.MRMLNodeNotFoundException:
            node = None
        return node
    else:
        return name

def getOrCreateNode(name, createFn=lambda n: None, performOnNode=lambda x: None, addToScene=True):
    node = getNode(name)
    if node is None:
        node = createFn(name)
        node.SetName(name)
        performOnNode(node)
        if addToScene:
            slicer.mrmlScene.AddNode(node)
    return node

def getOrCreateTransformNode(name):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLLinearTransformNode())

def getOrCreateVolumeNode(name):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLScalarVolumeNode())

def getOrCreateLabelMapVolumeNode(name):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLLabelMapVolumeNode())

def getOrCreateSegmentationNode(name):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLSegmentationNode())

def getOrCreateFiducialListNode(name, performOnNode=lambda x: None):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLMarkupsFiducialNode(), performOnNode=performOnNode)

def getOrCreateModelNode(name, performOnNode=lambda x: None):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLModelNode(), performOnNode=performOnNode)

def getOrCreateModelHierarchyNode(name, performOnNode=lambda x: None):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLModelHierarchyNode(), performOnNode=performOnNode)
    
def getOrCreateSubjectHierarchyFolder(name, performOnNode=lambda x: None):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLFolderDisplayNode(), performOnNode=performOnNode)

def getOrCreateMarkupsRoiNode(name, performOnNode=lambda x: None):
    return getOrCreateNode(name, createFn=lambda n: slicer.vtkMRMLAnnotationROINode(), performOnNode=performOnNode)

def deleteNodesIfExists(nameList):
    try:
      basestring
    except NameError:
      basestring = str
    # if only one string given, convert to list
    if isinstance(nameList, basestring):
        nameList = [nameList]
    for name in nameList:
        n = getNode(name)
        if n is not None:
            slicer.mrmlScene.RemoveNode(n)


'''
REGISTRATION
'''


def switchToTransformUI(selectedNode):
    # switch to transforms module and select trfT2hInit as transform
    slicer.util.selectModule("Transforms")
    selector = slicer.util.findChildren(name="TransformNodeSelector")[0]
    selector.setCurrentNode(selectedNode)


def setupInitialRegistration(nodeFixed, nodeMoving, trfMovingToFixedName):
    if nodeFixed is None or nodeMoving is None or not trfMovingToFixedName:
        print("Check your parameters @setupInitialRegistration!")
        raise ValueError("Invalid Parameters!")
    trf_parent = nodeFixed.GetParentTransformNode()

    # add init transform
    trf_init = getOrCreateTransformNode(trfMovingToFixedName)

    # get parent transform of nodeFixed space
    trf_parent_id = __GetID__(trf_parent)
    trf_init.SetAndObserveTransformNodeID(trf_parent_id)

    nodeMoving.SetAndObserveTransformNodeID(trf_init.GetID())
    setVolumeSelection(nodeFixed, nodeMoving)


def runAndApplyBRAINSRegistration(nodeFixed, nodeMoving, init_transform, output_transform, remove_init_trf=False):
    runBRAINSRegistration(nodeFixed, nodeMoving, nodeOutputTrf=output_transform, nodeTrfInit=init_transform)
    # replaceTransform(init_transform, output_transform)

    # delete node if desired....
    if remove_init_trf and init_transform is not None:
        deleteNodesIfExists(init_transform.GetName())


def runBRAINSRegistration(nodeFixed, nodeMoving, nodeOutputVol=None, nodeOutputTrf=None,
                          mode='Affine', samplingPercentage=0.005, nrBins=32, metric='MMI',
                          nodeTrfInit=None, nodeMaskFixed=None, nodeMaskMoving=None, interpolationMode='Linear'):
    # simplest example:
    # logic.runBRAINSRegistration(volT1,volT2,nodeOutputTrf=T2ToT1)
    regParameters = {}
    regParameters["fixedVolume"] = nodeFixed
    regParameters["movingVolume"] = nodeMoving
    regParameters["samplingPercentage"] = samplingPercentage
    regParameters["interpolationMode"] = interpolationMode
    regParameters["numberOfHistogramBins"] = nrBins
    regParameters["costMetric"] = metric
    if nodeMaskFixed is not None:
        regParameters["maskProcessingMode"] = 'ROI'
        regParameters["fixedBinaryVolume"] = nodeMaskFixed
    if nodeMaskMoving is not None:
        regParameters["maskProcessingMode"] = 'ROI'
        regParameters["movingBinaryVolume"] = nodeMaskMoving
    # initial transforms
    if nodeTrfInit is not None:
        regParameters["initialTransform"] = nodeTrfInit
    # transformation mode
    if mode == 'Affine':
        regParameters["useAffine"] = True
    elif mode == 'Rigid':
        regParameters["useRigid"] = True
    else:
        print('runBRAINSRegistration wrapper: Mode %s not yet implemented' % mode)
    # output options
    if nodeOutputTrf is not None:
        regParameters["linearTransform"] = nodeOutputTrf
    if nodeOutputVol is not None:
        regParameters["outputVolume"] = nodeOutputVol
    # slicer.util.selectModule('BRAINSFit')
    regBrains = slicer.modules.brainsfit
    slicer.cli.run(regBrains, None, regParameters, wait_for_completion=True)
    slicer.app.processEvents()

def registerBRAINSLinearInverse(nodeFixed, nodeMoving, nodeOutputVol=None, nodeOutputTrf=None,
                          mode='Affine', samplingPercentage=0.005, nrBins=32, metric='MMI',
                          nodeTrfInit=None, nodeMaskFixed=None, nodeMaskMoving=None, interpolationMode='Linear'):
    # nodeFixed should be the reference volume, nodeOutputTrf will move nodeMoving onto nodeFixed
    # nodeOutputTrf will be actually computed by registering nodeFixed onto nodeMoving and then taking the inverse
    # rationale:
    #   sometimes, it is better to register in an inverted manner
    #   example: I have volume A and another volume B which covers just a small
    #          slab of image B (e.g. A: T1, B: CISS, only in inner ear planes)
    #          and I want to register B to A. Then, in BRAINS, it is much better
    #          to register A to B (!), and invert the resulting matrix
    #          than registering A to B.

    # set up inverse transformation matrices nodeOutputTrfInv and nodeTrfInitInv
    nl = [nodeTrfInit,nodeOutputTrf]
    suffix_inv = '_inv'
    for trfNode in nl:
        if trfNode is not None:
            #print(trfNode.GetName())
            invertTransformNode(trfNode.GetName(), suffix=suffix_inv)

    # run BRAINS registration with fixed/moving nodes inverted
    nodeFixed_inv = nodeMoving
    nodeMoving_inv = nodeFixed
    nodeOutputTrf_inv = getOrCreateTransformNode(nodeOutputTrf.GetName()+suffix_inv)
    if nodeTrfInit is not None:
        nodeTrfInit_inv = getOrCreateTransformNode(nodeTrfInit.GetName()+suffix_inv)
    else:
        nodeTrfInit_inv = None
    nodeOutputVol = None # we don't want to interpolate the volume in the inverted way (it would have the wrong geometry/dimensions/position)
    runBRAINSRegistration(nodeFixed_inv, nodeMoving_inv, nodeOutputVol=nodeOutputVol, nodeOutputTrf=nodeOutputTrf_inv,
                          mode=mode, samplingPercentage=samplingPercentage, nrBins=nrBins, metric=metric,
                          nodeTrfInit=nodeTrfInit_inv, nodeMaskFixed=nodeMaskFixed, nodeMaskMoving=nodeMaskMoving, interpolationMode=interpolationMode)

    # copy the resulting transform to the output node nodeOutputTrf and Invert (!) the transformation
    nameOutputTrf = nodeOutputTrf.GetName()
    invertTransformNode(nameOutputTrf+suffix_inv, nameOutputTrf)

    # re-arrange transformations
    nodeFixed.SetAndObserveTransformNodeID(None)
    nodeMoving.SetAndObserveTransformNodeID(nodeOutputTrf.GetID())

    # CLEANUP!
    # delete transformations nodeOutputTrfInv and nodeTrfInitInv
    nl = [nodeTrfInit,nodeOutputTrf]
    nlnames = []
    for trfNode in nl:
        if trfNode is not None:
            nlnames.append(trfNode.GetName()+suffix_inv)
    deleteNodesIfExists(nlnames)

def registerBRAINSLocalInsideROI(nodeFixed, nodeMoving, nodeROI, nodeTrfOut, keepIntermediateNodes=False, mode='Affine'):
    listNodeNamesToDelete = []
    # crop volumes with ROI
    nodeFixedCrop  = cropVolumeWithRoi(nodeFixed, nodeROI, 'tempNodeFixed', interpolationMode='Linear',
                                      spacingScalingConstant=1.0,isotropic=True)
    nodeMovingCrop = cropVolumeWithRoi(nodeFixed, nodeROI, 'tempNodeMoving', interpolationMode='Linear',
                                      spacingScalingConstant=1.0,isotropic=True)
    listNodeNamesToDelete.append(nodeFixedCrop.GetName())
    listNodeNamesToDelete.append(nodeMovingCrop.GetName())
    # compute registration moving to fixed
    runBRAINSRegistration(nodeFixedCrop, nodeMovingCrop, nodeOutputVol=None,
                          nodeOutputTrf=nodeTrfOut, mode='Rigid')

    if not keepIntermediateNodes:
        deleteNodesIfExists(listNodeNamesToDelete)

def registerFiducials(nodeFidsFixed, nodeFidsMoving, nodeTrfOut, mode='Rigid', inplace=True):
    # def registerFiducials(nodeFidsFixed, nodeFidsMoving, nodeTrfOut, mode='Rigid', inplace=True)
    # registers nodeFidsMoving to nodeFidsFixed, stores transformation into nodeTrfOut
    # mode: can be 'Translation', 'Rigid', 'Similarity'
    # inplace: if True, fiducials will be used at their current world location (i.e. considering transformations)
    if inplace:
        arrFixed = slicer.util.arrayFromMarkupsControlPoints(nodeFidsFixed, world=True)
        arrMoving = slicer.util.arrayFromMarkupsControlPoints(nodeFidsMoving, world=True)
        nTmpFidsFixed = fiducialListFromArray('tmpFidsFixed', arrFixed)
        nTmpFidsMoving = fiducialListFromArray('tmpFidsMoving', arrMoving)
    else:
        nTmpFidsFixed = nodeFidsFixed
        nTmpFidsMoving = nodeFidsMoving
    # set up registration parameters
    regParameters = {}
    regParameters["fixedLandmarks"] = nTmpFidsFixed
    regParameters["movingLandmarks"] = nTmpFidsMoving
    regParameters["saveTransform"] = nodeTrfOut
    regParameters["transformType"] = mode
    # run CLI
    regFiducials = slicer.modules.fiducialregistration
    slicer.cli.run(regFiducials, None, regParameters, wait_for_completion=True)
    slicer.app.processEvents()
    # delete temporary data, if created
    if inplace:
        deleteNodesIfExists(['tmpFidsFixed','tmpFidsMoving'])


'''
UI STUFF UTILS
'''
def setSliceViewerLayers():
    print('setSliceViewerLayers is deprecated!')
    print('Use this new function instead:')
    print("slicer.util.setSliceViewerLayers(background='keep-current', foreground='keep-current', label='keep-current', foregroundOpacity=None, labelOpacity=None, fit=False)")

def setBackground(nodeVol):
    if isinstance(nodeVol, str):
        nodeVol = getNode(nodeVol)
    appLogic = slicer.app.applicationLogic()
    selectionNode = appLogic.GetSelectionNode()
    if nodeVol is None:
        selectionNode.SetReferenceActiveVolumeID(None)
    else:
        selectionNode.SetReferenceActiveVolumeID(nodeVol.GetID())
    appLogic.PropagateVolumeSelection()


def setForeground(nodeVol):
    if isinstance(nodeVol, str):
        nodeVol = getNode(nodeVol)
    appLogic = slicer.app.applicationLogic()
    selectionNode = appLogic.GetSelectionNode()
    if nodeVol is None:
        selectionNode.SetReferenceActiveVolumeID(None)
    else:
        selectionNode.SetReferenceSecondaryVolumeID(nodeVol.GetID())
    appLogic.PropagateVolumeSelection()


def setSegmentation(nodeVol):
    if isinstance(nodeVol, str):
        nodeVol = getNode(nodeVol)
    appLogic = slicer.app.applicationLogic()
    selectionNode = appLogic.GetSelectionNode()
    if nodeVol is None:
        selectionNode.SetReferenceActiveLabelVolumeID(None)
    else:
        selectionNode.SetReferenceActiveLabelVolumeID(nodeVol.GetID())
    appLogic.PropagateVolumeSelection()


def setOpacity(alpha=0.5):
    compositeViews = slicer.mrmlScene.GetNodesByClass('vtkMRMLSliceCompositeNode')
    compositeViews.InitTraversal()
    for i in range(compositeViews.GetNumberOfItems()):
        view = compositeViews.GetNextItemAsObject()
        view.SetForegroundOpacity(alpha)


def setVolumeSelection(bg=None, fg=None, opacity=0.5):
    # deprecated...
    # migrate to: slicer.util.setSliceViewerLayers(background='keep-current', foreground='keep-current', label='keep-current', foregroundOpacity=None, labelOpacity=None, fit=False)
    # works with volNode and volName
    if isinstance(bg, str):
        bg = getNode(bg)
    if isinstance(fg, str):
        fg = getNode(fg)
    setBackground(bg)
    setForeground(fg)
    setOpacity(alpha=opacity)

def setRenderVolume(nodeVol, nodeROI=None):
    if isinstance(nodeVol, str):
        nodeVol = getNode(nodeVol)
    volumeRenderingDisplayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(nodeVol)
    nodeVol.AddAndObserveDisplayNodeID(volumeRenderingDisplayNode.GetID())
    slicer.modules.volumerendering.logic().UpdateDisplayNodeFromVolumeNode(volumeRenderingDisplayNode, nodeVol)
    if nodeROI is not None:
        volumeRenderingDisplayNode.SetAndObserveROINodeID(nodeROI.GetID())
        volumeRenderingDisplayNode.CroppingEnabled = True
    # change visualization parameters in renderer
    #volumeRenderingPropertyNode = volumeRenderingDisplayNode.GetVolumePropertyNode()
    #colorTransferFunction = volumeRenderingPropertyNode.GetColor()
    #colorNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLProceduralColorNode')
    #colorNode.SetAndObserveColorTransferFunction(colorTransferFunction)
    #volumeNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

def visVol_VolumeRender(nodeVol, nodeROI=None):
    logic = slicer.modules.volumerendering.logic()
    displayNode = logic.CreateVolumeRenderingDisplayNode()
    displayNode.UnRegister(logic)
    slicer.mrmlScene.AddNode(displayNode)
    nodeVol.AddAndObserveDisplayNodeID(displayNode.GetID())
    logic.UpdateDisplayNodeFromVolumeNode(displayNode, nodeVol)
    if nodeROI is not None:
        displayNode.SetAndObserveROINodeID(nodeROI.GetID())
        displayNode.CroppingEnabled = True
        logic.UpdateDisplayNodeFromVolumeNode(displayNode, nodeVol)

def renderVolWithShift(volumeNode, shift=0.0):
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
    volRenWidget = slicer.modules.volumerendering.widgetRepresentation()
    if volRenWidget is None:
        logging.error('Failed to access volume rendering module')
        return
    # Make sure the proper volume property node is set
    volumePropertyNode = displayNode.GetVolumePropertyNode()
    if volumePropertyNode is None:
	    logging.error('Failed to access volume properties')
	    return
    volumePropertyNodeWidget = slicer.util.findChild(volRenWidget, 'VolumePropertyNodeWidget')
    volumePropertyNodeWidget.setMRMLVolumePropertyNode(volumePropertyNode)
    # Adjust the transfer function
    volumePropertyNodeWidget.moveAllPoints(shift, 0, False)

def sliceViewRecenter():
    appLogic = slicer.app.applicationLogic()
    appLogic.FitSliceToAll()

def setVolumeColorLUT(volNode, colorString):
    nColorList = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLColorTableNode', colorString)
    nColor = nColorList.GetItemAsObject(0)
    if nColor is None:
        print('dsgzValCissReg setVolumeColorLUT: There is no color node called %s' % colorString)
        return
    else:
        volNode.GetDisplayNode().SetAndObserveColorNodeID(nColor.GetID())

def zoom(factor,sliceNodes=None):
    """Zoom slice nodes by factor.
    factor: "Fit" or +/- amount to zoom
    sliceNodes: list of slice nodes to change, None means all.
    from:
    https://github.com/pieper/CompareVolumes/blob/master/CompareVolumes.py#L430-L448
    """
    if not sliceNodes:
        sliceNodes = getNodes('vtkMRMLSliceNode*')
    layoutManager = slicer.app.layoutManager()
    for sliceNode in sliceNodes.values():
        if factor == "Fit":
            sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
            if sliceWidget:
                sliceWidget.sliceLogic().FitSliceToAll()
        else:
            newFOVx = sliceNode.GetFieldOfView()[0] * factor
            newFOVy = sliceNode.GetFieldOfView()[1] * factor
            newFOVz = sliceNode.GetFieldOfView()[2]
            sliceNode.SetFieldOfView( newFOVx, newFOVy, newFOVz )
            sliceNode.UpdateMatrices()

def view3D_pitch(direction='up',n=1):
    lm = slicer.app.layoutManager()
    view = lm.threeDWidget(0).threeDView()
    if direction == 'up':
        view.pitchDirection = view.PitchUp
    elif direction == 'down':
        view.pitchDirection = view.PitchDown
    else:
        view.pitchDirection = view.PitchUp
    for i in range(n):
        view.pitch()

def view3D_yaw(direction='left',n=1):
    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    if direction == 'left':
        view.yawDirection = view.YawLeft
    elif direction == 'right':
        view.yawDirection = view.YawRight
    else:
        view.yawDirection = view.YawLeft
    for i in range(n):
        view.yaw()

def view3D_lookFromViewAxis(direction='right'):
    lm = slicer.app.layoutManager()
    view = lm.threeDWidget(0).threeDView()
    if direction.lower() == 'right':
        view.lookFromViewAxis(ctk.ctkAxesWidget.Right) 
    elif direction.lower() == 'left':
        view.lookFromViewAxis(ctk.ctkAxesWidget.Left) 
    elif direction.lower() == 'inferior':
        view.lookFromViewAxis(ctk.ctkAxesWidget.Inferior) 
    elif direction.lower() == 'superior':
        view.lookFromViewAxis(ctk.ctkAxesWidget.Superior) 
    elif direction.lower() == 'anterior':
        view.lookFromViewAxis(ctk.ctkAxesWidget.Anterior) 
    elif direction.lower() == 'posterior':
        view.lookFromViewAxis(ctk.ctkAxesWidget.Posterior) 

def zoom2D(factor,zoomRed=True,zoomGreen=True,zoomYellow=True):
    if zoomRed:
        sn = getSliceNode('Red')
        fov = sn.mrmlSliceNode().GetFieldOfView()
        sn.mrmlSliceNode().SetFieldOfView(fov[0]/factor,fov[1]/factor,fov[2])
    if zoomGreen:
        sn = getSliceNode('Green')
        fov = sn.mrmlSliceNode().GetFieldOfView()
        sn.mrmlSliceNode().SetFieldOfView(fov[0]/factor,fov[1]/factor,fov[2])
    if zoomYellow:
        sn = getSliceNode('Yellow')
        fov = sn.mrmlSliceNode().GetFieldOfView()
        sn.mrmlSliceNode().SetFieldOfView(fov[0]/factor,fov[1]/factor,fov[2])

def zoom3D(factor):
    camera = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCameraNode').GetCamera()
    camera.Zoom(factor)

def view3D_center():
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()

def threeDViewSetBackgroundColor(color):
    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    view.setBackgroundColor(qt.QColor(color))
    view.setBackgroundColor2(qt.QColor(color))
    view.forceRender()

def processEvents():
    slicer.app.processEvents()

def forceViewUpdate():
    slicer.app.processEvents()
    slicer.app.layoutManager().threeDWidget(0).threeDView().forceRender()

def getSliceNode(slicename):
    # this returns a node of class 'PythonQt.qMRMLWidgets.qMRMLSliceWidget'
    if slicename not in ['Red','Green','Yellow']:
        disp('slicerutil.getSliceNode: (slicename) - slicename must be one of: "Red"/"Green"/"Yellow".')
        return
    lm = slicer.app.layoutManager()
    sliceNode = lm.sliceWidget(slicename)
    return sliceNode

def setSliceOffset(slicename,offset,mode='offset'):
    # 
    sliceNode = getSliceNode(slicename)
    sn = sliceNode.mrmlSliceNode()
    # for vtkMRMLSliceNode nodes:
    if mode=='center':
        sn.JumpSliceByCentering(offset[0],offset[1],offset[2])
    elif mode=='offset':
        sn.JumpSliceByOffsetting(offset[0],offset[1],offset[2])
    else:
        disp('su.setSliceOffset: Mode "%s" unknown. Allowed modes are "offset" and "center".'%mode)

def setSliceOffsetToFiducial(nameFidList, fidName, 
                             mode='offset',
                             mode_override_green=None, 
                             mode_override_red=None, 
                             mode_override_yellow=None):
    # mode: Sets all slice jump modes at once
    # mode_override_green/red/yellow: Can individually override single-slice modes
    #                                 Can take values 'center/offset/keep'
    #                                 'keep' makes that slices jump not at all
    pt = arrayFromFiducialList(nameFidList,[fidName]).ravel()
    # deprecated: for qMRMLSliceWidget nodes:
    #setSliceOffset('Green',pt[1])
    #setSliceOffset('Red',pt[2])
    #setSliceOffset('Yellow',pt[0])
    # set the mode
    mode_green = mode_red = mode_yellow = mode
    # override where requested
    if mode_override_green is not None:
        mode_green = mode_override_green
    if mode_override_red is not None:
        mode_red = mode_override_red
    if mode_override_yellow is not None:
        mode_yellow = mode_override_yellow
    # offset unless "keep" as-is
    if mode_green!='keep':
        setSliceOffset('Green',pt,mode=mode_green)
    if mode_green!='keep':
        setSliceOffset('Red',pt,mode=mode_red)
    if mode_green!='keep':
        setSliceOffset('Yellow',pt,mode=mode_yellow)

def fitSliceViewToPoints(pointSourceName=None, sliceColor='Green', nr_pts_max=1000, seed=None, listFidNames=None):
    node_pts = slicer.util.getNode(pointSourceName)
    if node_pts is not None:
        node_type = node_pts.GetClassName()
        # Get point positions as numpy array
        if node_type == 'vtkMRMLModelNode':
            arr_pts = arrayFromModelWithTransform(pointSourceName)
            # random subsampling of model vertices, if model is too large (SVD would take too long)
            if nr_pts_max>0:
                if seed is not None:
                    np.random.seed(seed)
                arr_pts = arr_pts[np.random.choice(arr_pts.shape[0], nr_pts_max, replace=False), :]
        elif node_type == 'vtkMRMLMarkupsFiducialNode':
            arr_pts = arrayFromFiducialList(pointSourceName,listFidNames=listFidNames)
        else:
            print('fitSliceViewToPoints: only defined for models and fiducial lists but not for nodes of type: %s.'%node_type)
            return
        # Compute plane normal and positionlstsqPlaneEstimation(pts)
        planeNormal, planePosition, planeRotation = lstsqPlaneEstimation(arr_pts)
        planeNormal = planeNormal.ravel()
        planePosition = planePosition.ravel()
        planeX = planeRotation[:,0].ravel()
        # Re-orient slice markup node from scene
        sliceNode = slicer.app.layoutManager().sliceWidget(sliceColor).mrmlSliceNode()
        sliceNode.SetSliceToRASByNTP(planeNormal[0], planeNormal[1], planeNormal[2],
                                     planeX[0], planeX[1], planeX[2],
                                     planePosition[0], planePosition[1], planePosition[2], 0)        


def captureImageFromSliceView(offsets, pathwithprefix, slicename):
    # save a screenshot
    pattern = pathwithprefix + '_%s.png'
    if slicename not in ['Red','Green','Yellow']:
        disp('slicerutil.getSliceNode: (slicename) - slicename must be one of: "Red"/"Green"/"Yellow".')
        return

    widget = slicer.app.layoutManager().sliceWidget(slicename)
    view = widget.sliceView()
    logic = widget.sliceLogic()
    bounds = [0,]*6
    logic.GetSliceBounds(bounds)
    print(bounds)

    print(isinstance(offsets,list))

    if not isinstance(offsets, list):
        print('listifying...')
        offsets = [offsets]

    for offset in offsets:
        print(offset)
        #offset = bounds[4] + (step+1)/(1.*(steps+1)) * (bounds[5]-bounds[4])
        #print (step+1)/(1.*(steps+1))
        logic.SetSliceOffset(offset)
        view.forceRender()
        image = qt.QPixmap.grabWidget(view).toImage()
        image.save(pattern%(str(offset)))

def captureImageFromAllViews(filename):
    import ScreenCapture
    cap = ScreenCapture.ScreenCaptureLogic()
    cap.showViewControllers(False)
    cap.captureImageFromView(None,filename)
    cap.showViewControllers(True)

#su.captureImageFromSliceView([-11.0,-10.0,-9.0,-8.0,-7.0], '/Volumes/NEURO2TB_1/Grosshadern_50patient_MRI/resultsGroupAtlas2017/PPMPaper/testImg', 'Red')

'''
VOLUME UTILS
'''
def castVolume(volNodeNameIn,volNodeNameOut,type='Float'):
    # type can be: Char, UnsignedChar, Short, UnsignedShort,
    #              Int, UnsignedInt, Float, Double,
    nvolIn = getOrCreateVolumeNode(volNodeNameIn)
    nvolOut = getOrCreateVolumeNode(volNodeNameOut)
    params = {}
    params["InputVolume"] = nvolIn
    params["OutputVolume"] = nvolOut
    params["Type"] = type
    tmpmodule = slicer.modules.castscalarvolume
    slicer.cli.run(tmpmodule, None, params, wait_for_completion=True)

def cloneVolumeNode(nameSource, nameCloned):
    nSource = getOrCreateVolumeNode(nameSource)
    nCloned = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, nSource, nameCloned)
    return nCloned

def centerVolume(volNodeName):
    # CAUTION: removes the volume's location in the MRML scene and visualization (colormap/contrast etc.)
    img = sitku.PullVolumeFromSlicer(volNodeName)
    #print('Image size: %s'%str(img.GetSize()))
    #print('Image spacing: %s'%str(img.GetSpacing()))
    newCenter=-0.5*np.array(img.GetSize())*np.array(img.GetSpacing()) + 0.5*np.array(img.GetSpacing())
    #print('New center: %s'%str(newCenter))
    img.SetOrigin(tuple(newCenter.tolist()))
    sitku.PushVolumeToSlicer(img, name=volNodeName)

def centerVolumeTransform(volNodeName,nameTrf):
    nVol = getNode(volNodeName)
    # find trfVolumeToCentered via volume bounds
    b = [0.0,0.0,0.0,0.0,0.0,0.0]
    nVol.GetBounds(b)
    t = np.zeros(3,)
    t[0] = -1 * np.mean(b[0:2])
    t[1] = -1 * np.mean(b[2:4])
    t[2] = -1 * np.mean(b[4:6])
    M = transform4x4FromRandT(t=t)
    nTrfVolCentered = transformFrom4x4Matrix(nameTrf,M)
    return nTrfVolCentered

def visVol_SetThreshold(volNodeName, thresholds):
    nVol = getNode(volNodeName)
    if nVol is None:
        print('slicerutil.visVol_Contrast: Volume node with name "%s" does not exist in MRML tree.'%volNodeName)
        return
    nVol.GetDisplayNode().SetThreshold(thresholds[0],thresholds[1])
    nVol.GetDisplayNode().SetApplyThreshold(1)

def visVol_SetWindowLevel(volNodeName, windowlevel):
    nVol = getNode(volNodeName)
    if nVol is None:
        print('slicerutil.visVol_SetWindowLevel: Volume node with name "%s" does not exist in MRML tree.'%volNodeName)
        return
    nVol.GetDisplayNode().SetAutoWindowLevel(False)
    nVol.GetDisplayNode().SetWindowLevel(windowlevel[0],windowlevel[1])

def visVol_SetParams(volNodeName, thresholds, windowlevel):
    visVol_SetThreshold(volNodeName, thresholds)
    visVol_SetWindowLevel(volNodeName, windowlevel)

def visVol_SetContrastToRangeByString(volNodeName, rangeString, maskExcludeValue=None):
    nVol = getNode(volNodeName)
    if nVol is None:
        print('slicerutil.visVol_SetWindowLevel: Volume node with name "%s" does not exist in MRML tree.'%volNodeName)
    stats = volIntensityStats(volNodeName,maskExcludeValue=maskExcludeValue)
    if rangeString is None:
        print('slicerutil.visVol_SetContrastToRangeByString: no range string given')
        return
    if rangeString == 'UpperPercentile95':
        thresholds = [stats['perc95'],stats['max']]
        level = np.mean(np.array(thresholds))
        window = stats['max'] - level
        windowlevel = [window,level]
    if rangeString == 'MeanFullRange':
        thresholds = [stats['min'],stats['max']]
        level = np.mean(np.array(thresholds))
        window = stats['max'] - level
        windowlevel = [window,level]
    if rangeString == 'MedianPercentile5To95':
        thresholds = [stats['perc5'],stats['perc95']]
        level = stats['median']
        window = stats['perc95'] - stats['median']
        windowlevel = [window,level]
    if rangeString == 'MedianPercentile1To99':
        thresholds = [stats['perc1'],stats['perc99']]
        level = stats['median']
        window = stats['perc99'] - stats['median']
        windowlevel = [window,level]
    if rangeString == 'UpperHalf':
        thresholds = [stats['median'],stats['max']]
        level = np.mean(np.array(thresholds))
        window = stats['max'] - level
        windowlevel = [window,level]
    if rangeString == 'UpperQuartile':
        thresholds = [stats['perc75'],stats['max']]
        level = np.mean(np.array(thresholds))
        window = stats['max'] - level
        windowlevel = [window,level]
    if rangeString == 'CenteredAtZero':
        maxrange = np.max([np.abs(stats['min']), np.abs(stats['max'])])
        thresholds = [stats['min'],stats['max']]
        level = 0.0
        window = maxrange
        windowlevel = [window,level]
    visVol_SetThreshold(volNodeName, thresholds)
    visVol_SetWindowLevel(volNodeName, windowlevel)

def volIntensityStats(volNodeName,maskNodeName=None,maskExcludeValue=None,maskThresholdValue=None,
                      perctl_low=0.0, perctl_high=100.0):
    nVol = getNode(volNodeName)
    if nVol is None:
        print('slicerutil.visVol_Contrast: Volume node with name "%s" does not exist in MRML tree.'%volNodeName)
    if maskNodeName is not None:
        print('slicerutil.visVol_Contrast: feature maskNodeName not yet implemented')
    arr = slicer.util.arrayFromVolume(nVol)
    if maskExcludeValue is not None:
        vec = arr[arr!=maskExcludeValue]
    elif maskThresholdValue is not None:
        vec = arr[arr>maskThresholdValue]
    else:
        vec = arr.ravel()
    stats = {'min':np.min(vec)}
    stats['max'] = np.max(vec)
    stats['perc1'] = np.percentile(vec,1.0)
    stats['perc5'] = np.percentile(vec,5.0)
    stats['perc25'] = np.percentile(vec,25.0)
    stats['perc50'] = np.percentile(vec,50.0)
    stats['perc75'] = np.percentile(vec,75.0)
    stats['perc95'] = np.percentile(vec,95.0)
    stats['perc99'] = np.percentile(vec,99.0)
    stats['perc_low'] = np.percentile(vec,perctl_low)
    stats['perc_high'] = np.percentile(vec,perctl_high)
    stats['median'] = np.median(vec)
    stats['otsu'] = np_get_otsu_threshold(vec)
    return stats

def volNormalizeToRange(volNodeName, min, max, maskExcludeValue=None,maskThresholdValue=None):
    stats = volIntensityStats(volNodeName,maskExcludeValue=maskExcludeValue,maskThresholdValue=maskThresholdValue)
    volNode = getNode(volNodeName)
    range = max-min
    arr = slicer.util.arrayFromVolume(volNode)
    arr = (arr-min)/range
    arr[arr<0.0] = 0.0
    arr[arr>1.0] = 1.0
    slicer.util.updateVolumeFromArray(volNode, arr)
    
def volNormalizeRobust(volNodeName, 
                       perctl_low=0.0, perctl_high=100.0, 
                       out_low=0.0, out_high=1.0, 
                       maskExcludeValue=None,maskThresholdValue=None,
                       clip_low=True, clip_high=True):
    stats = volIntensityStats(volNodeName,
                              maskExcludeValue=maskExcludeValue,
                              maskThresholdValue=maskThresholdValue,
                              perctl_low=perctl_low,
                              perctl_high=perctl_high)
    # volume node and array
    volNode = getNode(volNodeName)
    arr = slicer.util.arrayFromVolume(volNode)
    # scale intensities
    imin = stats['perc_low']
    imax = stats['perc_high']
    irange = imax - imin
    arr = (arr - imin) / irange
    # scale to out_range
    irange_out = out_high - out_low
    arr = arr * irange_out
    arr = arr + out_low
    # clip to range
    if clip_low:
        arr[arr<out_low] = out_low
    if clip_high:
        arr[arr>out_high] = out_high
    slicer.util.updateVolumeFromArray(volNode, arr)

def volBinarizeWithThreshold(volNodeName, threshold):
    volNode = getNode(volNodeName)
    arr = slicer.util.arrayFromVolume(volNode)
    arr[arr<=threshold] = 0.0
    arr[arr>threshold] = 1.0
    slicer.util.updateVolumeFromArray(volNode, arr)
    
def volClipIntensities(volNodeName, clipLow=None, clipHigh=None):
    volNode = getNode(volNodeName)
    arr = slicer.util.arrayFromVolume(volNode)
    arr[arr<=clipLow] = clipLow
    arr[arr>clipHigh] = clipHigh
    slicer.util.updateVolumeFromArray(volNode, arr)

def volInvertIntensities(volNodeName):
    # useful for volume rendering of e.g. CT
    volNode = getNode(volNodeName)
    arr = slicer.util.arrayFromVolume(volNode)
    imin = np.min(arr.ravel())
    imax = np.max(arr.ravel())
    arr = imax-arr
    slicer.util.updateVolumeFromArray(volNode, arr)

def volReplaceValue(volNodeName,valueOld,valueNew):
    nVol = getNode(volNodeName)
    if nVol is None:
        print('slicerutil.visVol_Contrast: Volume node with name "%s" does not exist in MRML tree.'%volNodeName)
    img = sitku.PullFromSlicer(nVol)
    arr = sitk.GetArrayFromImage(img)
    arr[arr==valueOld] = valueNew
    slicer.util.updateVolumeFromArray(nVol,arr)

def volConvertToLabelmap(nameVol):
    vl = slicer.modules.volumes.logic()
    vn = getNode(nameVol)
    lmn = vl.CreateAndAddLabelVolume(vn,vn.GetName()+'_convertedToLabelmap')
    vl.CreateLabelVolumeFromVolume(slicer.mrmlScene, lmn, vn)
    deleteNodesIfExists(vn.GetName())
    lmn.SetName(lmn.GetName().replace('_convertedToLabelmap',''))
    return lmn
    
def volCreateFromScratch(name='MyNewVolume',
                         size=[128,128,128],
                         type=vtk.VTK_UNSIGNED_CHAR,
                         origin=[0.0, 0.0, 0.0],
                         spacing=[1.0, 1.0, 1.0],
                         directions=[[1,0,0], [0,1,0], [0,0,1]],
                         fillValue=0,
                         flagCentered=True
                         ):
    nodeName = name
    imageSize = size
    voxelType = type
    fillVoxelValue = fillValue
    if flagCentered:
        newOrigin = np.array(origin) - 0.5*(np.array(size)*np.array(spacing))
        imageOrigin = newOrigin.tolist()
        imageSpacing = spacing
        imageDirections = directions
    else:
        imageOrigin = origin
        imageSpacing = spacing
        imageDirections = directions
    # Create an empty image volume, filled with fillVoxelValue
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(imageSize)
    imageData.AllocateScalars(voxelType, 1)
    thresholder = vtk.vtkImageThreshold()
    thresholder.SetInputData(imageData)
    thresholder.SetInValue(fillVoxelValue)
    thresholder.SetOutValue(fillVoxelValue)
    thresholder.Update()
    # Create volume node
    volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
    volumeNode.SetOrigin(imageOrigin)
    volumeNode.SetSpacing(imageSpacing)
    volumeNode.SetIJKToRASDirections(imageDirections)
    volumeNode.SetAndObserveImageData(thresholder.GetOutput())
    volumeNode.CreateDefaultDisplayNodes()
    volumeNode.CreateDefaultStorageNode()
    return volumeNode

'''
LABELMAP UTILS
'''
def labelmapJoinBinarySegmentations(nameLabelmapOut,listLabelmapNames,listNewIndices=None):
    nTgt = getNode(nameLabelmapOut)
    if nTgt is None:
        # clone on of the input labelmaps, we will overwrite the clone's np array
        print(listLabelmapNames[0])
        nTgt = cloneNodeInSubjectHierarchy(listLabelmapNames[0], nameLabelmapOut)
    # labelmaps in listLabelmapNames need to have the same voxel grid already
    arrTgt = slicer.util.arrayFromVolume(nTgt)
    arrTgt = arrTgt.copy()*0 # erase all labels so far
    for idx, nameLab in enumerate(listLabelmapNames):
        nLab = getNode(nameLab)
        print(nLab.GetName())
        arrLab = slicer.util.arrayFromVolume(nLab)
        if listNewIndices is not None:
            idxNew = listNewIndices[idx]
        else:
            idxNew = idx+1
        arrTgt[arrLab==1] = idxNew#(idxNew*np.ones(arrTgt.shape))[arrLab==1]
    nTgt.SetAndObserveImageData(vtk.vtkImageData())
    slicer.util.updateVolumeFromArray(nTgt, arrTgt)
    return nTgt

'''
CROP/RESAMPLE VOLUMES
'''
def cropVolumeWithRoi(nVol, nRoi, outName=None, interpolationMode='Linear', spacingScalingConstant=1.0,
                      isotropic=True, targetResolution=0.0, keepIntermediateNodes=False):
    ''' 
    nVol, nRoi: actual nodes, not node names
    Possible interpolations: Linear|Nearest|Sinc|Bspline
    '''
    def toString(num):
        if int(num) == num:
            strOut = str(int(num))
        else:
            strOut = ('%0.1f' % num)
        return strOut

    # create name of output node: if already exists, delete first!
    # otherwise, many identically named subvolumes will be created
    # same if this volume was cropped before already
    if outName is None:
        outName = nVol.GetName() + '-subvolume-scale_' + toString(spacingScalingConstant)
    #deleteNodesIfExists([outNameDefault, outName])
    outputVolume = getOrCreateVolumeNode(outName)

    # if a target resolution is given, calculate the spacingScalingConstant
    if targetResolution != 0.0:
        resolutionMin = min( nVol.GetSpacing() )
        spacingScalingConstant = targetResolution/resolutionMin

    # use crop volume module to perform this operation
    cropVolumeNode = slicer.vtkMRMLCropVolumeParametersNode()
    cropVolumeNode.SetScene(slicer.mrmlScene)
    cropVolumeNode.SetName('CropVolume_node')
    cropVolumeNode.SetIsotropicResampling(isotropic)
    cropVolumeNode.SetSpacingScalingConst(spacingScalingConstant)
    cropVolumeNode.SetOutputVolumeNodeID(outputVolume.GetID())
    if interpolationMode == 'Nearest':
        cropVolumeNode.SetInterpolationMode(0)
    elif interpolationMode == 'Linear':
        cropVolumeNode.SetInterpolationMode(1)
    elif interpolationMode == 'Sinc':
        cropVolumeNode.SetInterpolationMode(2)
    elif interpolationMode == 'Bspline':
        cropVolumeNode.SetInterpolationMode(3)
    else:
        print(
            'interpolationMode "%s" not defined. Possible values are "Nearest/Linear/Sinc/Bspline"' % interpolationMode)
    slicer.mrmlScene.AddNode(cropVolumeNode)
    cropVolumeNode.SetInputVolumeNodeID(nVol.GetID())
    cropVolumeNode.SetROINodeID(nRoi.GetID())
    cropVolumeLogic = slicer.modules.cropvolume.logic()
    cropVolumeLogic.Apply(cropVolumeNode)
    slicer.app.processEvents()
    outputVolume.SetName(outName)
    return outputVolume


def resampleVolumeWithReference(inputVolume, referenceVolume, outputVolume,
                                trfLinear=None,
                                interpolationMode='linear'):
    # interpolationMode \in: 'linear', 'nn', 'ws', 'bs'
    params = {}
    params["inputVolume"] = inputVolume
    params["referenceVolume"] = referenceVolume
    params["outputVolume"] = outputVolume
    if trfLinear is not None:
        params["transformationFile"] = trfLinear
    else:
        # create an identity transform node
        print('WARNING: slicerutil.resampleVolumeWithReference: trfLinear not given, assuming identity transform')
        trfLinear = getOrCreateTransformNode('trfIdentity')
        params["transformationFile"] = trfLinear
    if not any(np.array(['linear', 'nn', 'ws', 'bs']) == interpolationMode):
        print(
            'resampleVolumeWithReference: interpolationMode %s not valid, allowed modes are "linear/nn/ws/bs"' % interpolationMode)
        return
    else:
        params["interpolationType"] = interpolationMode
    resampleVol = slicer.modules.resamplescalarvectordwivolume
    #print(params)
    slicer.cli.run(resampleVol, None, params, wait_for_completion=True)

def resampleScalarVolumeResolution(inputVolNode,outputVolNode,newResolution,interpolation='linear'):
    # interpolation can be: {linear}/nearestNeighbor/bspline/hamming/cosine/welch/lanczos/blackman
    # newResolution like [1.0,1.0,1.0]
    params = {}
    params["outputPixelSpacing"] = newResolution
    params["interpolationType"] = interpolation
    params["InputVolume"] = inputVolNode
    params["OutputVolume"] = outputVolNode
    resampleVol = slicer.modules.resamplescalarvolume
    slicer.cli.run(resampleVol, None, params, wait_for_completion=True)
    
def n4normalization(inputVolName,outputVolName,maskVolName=None,biasFieldName=None,initialMeshResolution=[1,1,1]):
    params = {}
    params["inputImageName"] = getOrCreateVolumeNode(inputVolName)
    params["outputImageName"] = getOrCreateVolumeNode(outputVolName)
    if maskVolName is not None:
        params["maskImageName"] = getOrCreateVolumeNode(maskVolName)
    if biasFieldName is not None:
        params["outputBiasFieldName"] = getOrCreateVolumeNode(biasFieldName)
    if initialMeshResolution is not None:
        params["initialMeshResolution"] = initialMeshResolution
    n4norm = slicer.modules.n4itkbiasfieldcorrection
    slicer.cli.run(n4norm, None, params, wait_for_completion=True)


def maskVolumeWithLabelVolume(nVol,nSeg):
    # TODO: let user choose labels in nSEG to create a mask from
    # right now, only nSeg>0

    # images
    imgVol = sitku.PullFromSlicer(nVol)
    imgSeg = sitku.PullFromSlicer(nSeg)

    # arrays
    arrVol = sitk.GetArrayFromImage(imgVol)
    arrSeg = sitk.GetArrayFromImage(imgSeg)

    # manipulate
    arrVol[arrSeg<75] = 0.0

    # update nVol
    slicer.util.updateVolumeFromArray(nVol,arrVol)

'''
SEGMENT EDITOR UTILS 
'''
def segmentationGetSegmentIDByName(nameSeg,nameRegion):
    nseg = getNode(nameSeg)
    if nseg is not None:
        segID = nseg.GetSegmentation().GetSegmentIdBySegmentName(nameRegion)
    else:
        segID = ''
    return segID

def segmentationAddSegmentByName(nameSeg,nameRegion):
    nseg = getNode(nameSeg)
    if nseg is not None:
        segID = nseg.GetSegmentation().GetSegmentIdBySegmentName(nameRegion)
        if segID=='':
            nseg.GetSegmentation().AddEmptySegment(nameRegion)
            segID = nseg.GetSegmentation().GetSegmentIdBySegmentName(nameRegion)
        else:
            print('WARNING (segmentationAddSegmentByName): Segment %s already present in segmentation %s.\nSkipped adding an empty segment with same name.'%(nameRegion,nameSeg))
    else:
        segID = ''
    return segID

def segmentationListRegions(nameSeg):
    nseg = getNode(nameSeg)
    seg = nseg.GetSegmentation()
    a = vtk.vtkStringArray()
    seg.GetSegmentIDs(a)
    list_names = []
    list_ids = []
    for i in range(a.GetNumberOfValues()):
        list_ids.append(a.GetValue(i))
        list_names.append(seg.GetSegment(a.GetValue(i)).GetName())
    return list_names, list_ids

def segmentationGetCenterOfMassByRegionName(nameSeg, nameRegion):
    # convert nameRegion to a list
    if isinstance(nameRegion, str):
        nameRegion = [nameRegion]
    nrRegions = len(nameRegion)
    # extract center-of-mass coordinates c of each region into a matrix C
    C = []
    n = getNode(nameSeg) 
    s = n.GetSegmentation()
    for name in nameRegion:
        print(name)
        ss = s.GetSegment(s.GetSegmentIdBySegmentName(name))
        pd = ss.GetRepresentation('Closed surface') # vtkPolyData (surface mesh) representation
        if pd is None:
            ss.AddRepresentation('Closed surface',vtk.vtkPolyData())
            pd = ss.GetRepresentation('Closed surface')
        com = vtk.vtkCenterOfMass()
        com.SetInputData(pd)
        com.Update()
        c = com.GetCenter()
        print(c)
        C.append(c)
    return np.array(C)

def labelmapGetCenterOfMassByRegionIndices(nameLabelmap, listIdxRegion=[1]):
    if isinstance(listIdxRegion,int):
        listIdxRegion = [listIdxRegion]
    nlab = getNode(nameLabelmap)
    labimg = sitku.PullFromSlicer(nlab)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labimg)
    C = []
    for idx in listIdxRegion:
        c = stats.GetCentroid(idx)
        C.append(c)
    return np.array(C)

def segmentationRegionJumpSlices(nameSeg, nameRegion):
    nseg = getNode(nameSeg)
    segID = nseg.GetSegmentation().GetSegmentIdBySegmentName(nameRegion)
    if segID != '':
        centroid = [0.0,0.0,0.0]; 
        nseg.GetSegmentCenterRAS(segID,centroid)
        markupsLogic = slicer.modules.markups.logic()
        markupsLogic.JumpSlicesToLocation(centroid[0],centroid[1],centroid[2], False)
        
def segmentationCollapseLabels(nameSeg, nameRegion):
    # segmentation node
    #ns = getNode(nameSeg)
    # segmentation itself
    #nss.AddEmptySegment(nameRegion)
    print('"su.segmentationCollapseLabels" not implemented.')
    
def segmentationExportToLabelmapByRegionNames(nameSeg, nameRefVol, nameLabelmap, listSegmentNames, createSegmentIfMissing=True):
    # segmentation and vol nodes
    nseg = getNode(nameSeg)
    nvol = getNode(nameRefVol)
    # segment names
    snames = listSegmentNames 
    nlab = getOrCreateLabelMapVolumeNode(nameLabelmap)
    for idx,sname in enumerate(snames):
        print('Converting region %s to labelmap.'%sname)
        sid = nseg.GetSegmentation().GetSegmentIdBySegmentName(sname)
        # in case a region with the requested name does not exist, we add an empty region to the segmentation
        if sid=='':
            if createSegmentIfMissing:
                nseg.GetSegmentation().AddEmptySegment(sname)
                sid = nseg.GetSegmentation().GetSegmentIdBySegmentName(sname)
            else:
                print('Segment %s does not exist in segmentation node %s. Exiting, returning None.'%(sname, nameSeg))
                return None
        # list of region names needs to be converted into a vector vtkStringArray 
        strarr = vtk.vtkStringArray()
        strarr.InsertNextValue(sid)
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(nseg, strarr, nlab, nvol)
    return nlab

def segmentationExportToModelsByRegionNames(nameSeg, nameModelHierarchyFolderToCreate, listSegmentNames, verbose=False):
    # segmentation and vol nodes
    nseg = getNode(nameSeg)
    # segment names
    snames = listSegmentNames # ['ILR','SLR','MLR','SLL','GGO','ILL','CON']
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), nameModelHierarchyFolderToCreate)
    for idx,sname in enumerate(snames):
        if verbose:
            print('Converting region %s to model.'%sname)
        sid = nseg.GetSegmentation().GetSegmentIdBySegmentName(sname)
        # in case a region with the requested name does not exist, we add an empty region to the segmentation
        if sid=='':
            nseg.GetSegmentation().AddEmptySegment(sname)
            sid = nseg.GetSegmentation().GetSegmentIdBySegmentName(sname)
        # list of region names needs to be converted into a vector vtkStringArray 
        strarr = vtk.vtkStringArray()
        strarr.InsertNextValue(sid)
        #slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToModels(nseg, strarr, nmodhier)
        slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToModels(nseg, strarr,exportFolderItemId)
    return exportFolderItemId

def segmentationImportModel(nameSeg, nameModel, nameSegment=''):
    nseg = getNode(nameSeg)
    nmod = getNode(nameModel)
    success = slicer.vtkSlicerSegmentationsModuleLogic.ImportModelToSegmentationNode(nmod, nseg, '')
    if nameSegment != '':
        sid = nseg.GetSegmentation().GetSegmentIdBySegmentName(nameModel)
        nseg.GetSegmentation().GetSegment(sid).SetName(nameSegment)

def segmentationImportLabelmap(nameSeg, nameLabelmap, nameSegment=''):
    nseg = getNode(nameSeg)
    nlab = getNode(nameLabelmap)
    success = slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(nlab, nseg, '')
    if nameSegment != '':
        sid = nseg.GetSegmentation().GetSegmentIdBySegmentName(nameLabelmap)
        nseg.GetSegmentation().GetSegment(sid).SetName(nameSegment)

def segmentationCreateClosedSurfaceRepresentation(nameSeg):
    getNode(nameSeg).CreateClosedSurfaceRepresentation()

def segmentationSetReferenceImageGeometryParameterFromVolumeNode(nameSeg, nameVol):
    nvol = getNode(nameVol)
    nseg = getNode(nameSeg)
    nseg.SetReferenceImageGeometryParameterFromVolumeNode(nvol)

def segmentationSetVisualization(nameSeg, opacity2D=None, opacity3D=None,
                                 show2DFill=None, show2DOutline=None,
                                 visibility3D=None, visibilityAllSegments=None):
    # TODO: set single segment visibilities:
    # nsegdisp.SetSegmentVisibility(nseg.GetSegmentation().GetSegmentIdBySegmentName('whole'), False)
    nseg = getNode(nameSeg)
    nsegdisp = nseg.GetDisplayNode()
    nsegdisp.SetAllSegmentsOpacity3D(0.5)
    if show2DFill is not None:
        nsegdisp.SetAllSegmentsVisibility2DFill(show2DFill)
    if show2DOutline is not None:
        nsegdisp.SetAllSegmentsVisibility2DOutline(show2DOutline)
    if opacity2D is not None:
        nsegdisp.SetAllSegmentsOpacity(opacity2D)
    if opacity3D is not None:
        nsegdisp.SetOpacity3D(opacity3D)
    if visibilityAllSegments is not None:
        nsegdisp.SetAllSegmentsVisibility(visibilityAllSegments)
    if visibility3D is not None:
        nsegdisp.SetVisibility3D(visibility3D)

def segmentationSetVisualizationSingleSegment(nameSegmentation, nameSegment, color=None, visibility=None):
    nseg = getNode(nameSegmentation)
    segid = segmentationGetSegmentIDByName(nameSegmentation,nameSegment)
    if visibility is not None:
        nseg.GetDisplayNode().SetSegmentVisibility(segid,visibility)
    if color is not None:
        nseg.GetSegmentation().GetSegment(segid).SetColor(color[0],color[1],color[2])

def segmentationComputeOverlapMeasures(name_segnode_ref, name_vol_ref,
                                       name_segment_ref, name_segment_cmp, 
                                       name_segnode_cmp=None):
    nseg_ref = getNode(name_segnode_ref)
    if name_segnode_cmp is not None:
        nseg_cmp = getNode(name_segnode_cmp)
    else:
        nseg_cmp = nseg_ref
        name_segnode_cmp = name_segnode_ref
    segid_ref = segmentationGetSegmentIDByName(name_segnode_ref,name_segment_ref)
    segid_cmp = segmentationGetSegmentIDByName(name_segnode_cmp,name_segment_cmp)
    nlab_ref = segmentationExportToLabelmapByRegionNames(name_segnode_ref, name_vol_ref, 'tmp_labelmap_ref', [name_segment_ref], createSegmentIfMissing=False)
    nlab_cmp = segmentationExportToLabelmapByRegionNames(name_segnode_cmp, name_vol_ref, 'tmp_labelmap_cmp', [name_segment_cmp], createSegmentIfMissing=False)
    # convert to ITK images
    img_ref = sitku.PullFromSlicer(nlab_ref)
    img_cmp = sitku.PullFromSlicer(nlab_cmp)
    # compute quality measures
    q = dict()
    # compute overlap measures
    lof = sitk.LabelOverlapMeasuresImageFilter()
    lof.Execute(img_ref,img_cmp)
    q['dice'] = lof.GetDiceCoefficient()
    q['jaccard'] = lof.GetJaccardCoefficient()
    q['fnr'] = lof.GetFalseNegativeError()
    q['fpr'] = lof.GetFalsePositiveError()
    #q['overlap_mean'] = lof.GetMeanOverlap()
    #q['overlap_union'] = lof.GetUnionOverlap()
    #q['volume_similarity'] = lof.GetVolumeSimilarity()
    # compute surface distance measures
    sdf = sitk.HausdorffDistanceImageFilter()
    sdf.Execute(img_ref,img_cmp)
    q['hausdorff'] = sdf.GetHausdorffDistance()
    q['hausdorff_avg'] = sdf.GetAverageHausdorffDistance()
    # cleanup
    deleteNodesIfExists([nlab_ref.GetName(), nlab_cmp.GetName()])
    # done
    return q

'''
FIDUCIAL LISTS / INTERACTIVE MODE
'''
def labelIndexInFidList(nameFidList,label):
    fl = getNode(nameFidList)
    if fl is not None:
        nrfids = fl.GetNumberOfFiducials()
        labels = []
        for i in range(nrfids):
            labels.append(fl.GetNthFiducialLabel(i))
        idxs = np.where(np.array(labels)==label)
        if idxs[0].shape[0]>0:
            idx = np.where(np.array(labels)==label)[0][0]
        else:
            idx = -1
        return idx
    else:
        return -1

def createFidLists(cfl_name_prefix, cfl_name_fids):
    mainList = getOrCreateFiducialListNode(cfl_name_prefix + "MAIN", performOnNode=lambda x: x.Reset(None))

    def reloadMainList(rml_name_prefix, rml_name_fids):
        # get and reset main list
        mainList = getOrCreateFiducialListNode(rml_name_prefix + "MAIN")
        mainList.RemoveAllMarkups()

        # retrieve element from every sublist (syntax: PREFIX_SUBLIST)
        for fid_name in rml_name_fids:
            fid_i = getOrCreateFiducialListNode(rml_name_prefix + "_" + fid_name)
            # copy fiducial into main list if available
            if fid_i.GetNumberOfFiducials() == 1:
                worldCoords = 4 * [0.0]
                fid_i.GetNthFiducialWorldCoordinates(0, worldCoords)
                main_i = mainList.AddFiducial(0.0, 0.0, 0.0, fid_name)
                mainList.SetNthFiducialWorldCoordinates(main_i, worldCoords)

    for fid_index, fidName in enumerate(cfl_name_fids):
        # create sublist for each fiducial
        fid_i = getOrCreateFiducialListNode(cfl_name_prefix + "_" + fidName, performOnNode=lambda x: x.Reset(None))

        def markupCallback(mcb_name_prefix, mcb_name_fids, index, caller_fidList, caller_eventId):
            # remove all elements except the last
            for i in range(caller_fidList.GetNumberOfFiducials() - 1):
                caller_fidList.RemoveMarkup(0)
            # rename fid element
            if caller_fidList.GetNumberOfFiducials() == 1:
                caller_fidList.SetNthFiducialLabel(0, mcb_name_fids[index])
            # reload main list
            partial(reloadMainList, mcb_name_prefix, mcb_name_fids)()

        # call markupCallback on any modification
        fid_i.AddObserver("ModifiedEvent", partial(markupCallback, cfl_name_prefix, cfl_name_fids, fid_index))


def addFiducialInteractively(name_prefix, name_fid):
    fid = getOrCreateFiducialListNode(name_prefix + "_" + name_fid)
    if fid is not None:
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(fid.GetID())
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
    else:
        print(
            "Something went wrong during AddFiducialInteractively! Node %s was not found", name_prefix + "_" + name_fid)
            
def addRoiInteractively(name_roi, pos=None, radius=None):
    roi = getOrCreateMarkupsRoiNode(name_roi)
    if roi is not None:
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLAnnotationROINode")
        selectionNode.SetActivePlaceNodeID(roi.GetID())
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        if pos is not None:
            roi.SetControlPointWorldCoordinates(0, pos[0:3], 0, 0)
        if radius is not None:
            roi.SetControlPointWorldCoordinates(1, radius[0:3], 0, 0)
        return roi
    else:
        print(
            "Something went wrong during addRoiInteractively! Node %s was not found", name_roi)

def getFiducialPosition(nameFidList, nameFid):
    fids = getNode(nameFidList)
    if fids is None:
        print('Fiducial list does not exist.')
        return [0.0,0.0,0.0]#,1.0]
    else:
        idx = labelIndexInFidList(nameFidList, nameFid)
        if idx==-1:
            print('There is no fiducial with label "%s" in fiducial markups list "%s"'%(nameFid,nameFidList))
            return [0.0,0.0,0.0,1.0]
        pos=[0.0,0.0,0.0,0.0]
        fids.GetNthFiducialWorldCoordinates(idx,pos)
        return pos[0:3]


def fiducialListFromArray(nameFidList, arr, listFidNames=None):
    nfids = getOrCreateFiducialListNode(nameFidList)
    nrfids = arr.shape[0]
    if listFidNames is None:
        listFidNames = ['%s-%d'%(nfids.GetName(),i+1) for i in range(nrfids)]
    for i in range(nrfids):
        nfids.AddFiducial(arr[i,0], arr[i,1], arr[i,2], listFidNames[i])
    return nfids
    
        
def arrayFromFiducialList(nameFidList, listFidNames=None):
    # if listFidNames is given, only those fiducial positions are extracted
    # otherwise, all fiducial positions in the list are converted to an array
    fids = getOrCreateFiducialListNode(nameFidList)
    if listFidNames is None:
        n = fids.GetNumberOfFiducials()
        P = []
        for i in range(n):
            pos = [0.0,0.0,0.0,0.0]
            fids.GetNthFiducialWorldCoordinates(i, pos)
            P.append(pos)
    else:
        n = len(listFidNames)
        P = []
        for i in range(n):
            pos = [0.0,0.0,0.0,0.0]
            pos = getFiducialPosition(nameFidList, listFidNames[i])
            P.append(pos)
    return np.array(P)
    
def visFid_SetVisibility(fids, 
                         visibility=None,
                         textScale=None,
                         color=None,
                         glyph_scale=None):
    if isinstance(fids,str):
        nfid = getNode(fids)
    else:
        nfid = fids
    dn = nfid.GetDisplayNode()
    if visibility is not None:
        nrFids = nfid.GetNumberOfFiducials()
        for i in range(nrFids):
            nfid.SetNthFiducialVisibility(i,visibility)
    if textScale is not None:
        dn.SetTextScale(textScale)
    if color is not None:
        dn.SetSelectedColor(color)
    if glyph_scale is not None:
        dn.SetGlyphScale(glyph_scale)
   
def fiducialDistance(nameFidList, listFidNames):
    return np.linalg.norm(np.diff(arrayFromFiducialList(nameFidList,listFidNames),axis=0))
   
'''
OTHER MARKUP UTILS
'''
def getOrCreateRoiAtPosition(name, pos=[0,0,0], radius=[1,1,1]):
    '''
        pos, radius ... 3-or-more element vector (list or np.array)
    '''
    #print('bing: %s' % name)
    nRoi = getOrCreateMarkupsRoiNode(name)
    nRoi.SetControlPointWorldCoordinates(0, pos[0:3], 0, 0)
    nRoi.SetControlPointWorldCoordinates(1, radius[0:3], 0, 0)
    return nRoi

'''
MODEL STUFF
'''
# get model statistics
def modelStatistics(nameModel):
    # get model node
    nodeModel = slicer.util.getNode(nameModel)
    # get points array
    arr = slicer.util.arrayFromModelPoints(nodeModel)
    
    stats = dict()
    stats['nrVertices'] = arr.shape[0]
    stats['centerOfMass'] = np.mean(arr,axis=0)
    return stats


# get model points under current transformation
def arrayFromModelWithTransform(nameModel):
    # get model node
    nodeModel = slicer.util.getNode(nameModel)
    # if this model is under a transform:
    if nodeModel.GetParentTransformNode() is not None:
        # clone model 
        tempModelName = '_todelete_tempModel'
        nodeCloned = cloneNodeInSubjectHierarchy(nameModel, tempModelName)
        # harden cloned model
        hardenTransformOfNode(tempModelName)
        # get array
        arr = slicer.util.arrayFromModelPoints(nodeCloned)
        # remove cloned model
        deleteNodesIfExists([tempModelName])
    else:
        arr = slicer.util.arrayFromModelPoints(nodeCloned)
    # return array
    return arr

# Update the sphere from the fiducial points
def drawSphereAtLoc(centerPointCoord,radius,name=None,
                    color=None,
                    opacity=None,
                    visibility=None,
                    intersectionThickness=None,
                    intersectionOpacity=None,
                    intersectionVisibility=False):
    # from: https://www.slicer.org/wiki/Documentation/Nightly/ScriptRepository
    # that version supports drawing a sphere between two points
    #import math
    # Get markup node from scene
    #markups=getNode('F')
    #centerPointCoord = [0.0, 0.0, 0.0]
    #markups.GetNthFiducialPosition(0,centerPointCoord)
    #circumferencePointCoord = [0.0, 0.0, 0.0]
    #markups.GetNthFiducialPosition(1,circumferencePointCoord)
    #radius=math.sqrt((centerPointCoord[0]-circumferencePointCoord[0])**2+(centerPointCoord[1]-circumferencePointCoord[1])**2+(centerPointCoord[2]-circumferencePointCoord[2])**2)
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(centerPointCoord)
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(30)
    sphere.SetThetaResolution(30)
    sphere.Update()
    # Create model node and add to scene
    modelsLogic = slicer.modules.models.logic()
    model = modelsLogic.AddModel(sphere.GetOutput())
    if name is None:
        name = 'sphere'
    model.SetName(name)
    model.GetDisplayNode().SetSliceIntersectionVisibility(True)
    model.GetDisplayNode().SetSliceIntersectionThickness(3)
    model.GetDisplayNode().SetColor(1,1,0)
    # change visualization
    modelChangeVisualization(name,color=color,
                    opacity=opacity,
                    visibility=visibility,
                    intersectionThickness=intersectionThickness,
                    intersectionOpacity=intersectionOpacity,
                    intersectionVisibility=intersectionVisibility)
    return model

def drawArrow(start,end,
                    name=None,
                    color=(1.0,0.0,0.0),
                    opacity=None,
                    visibility=None,
                    thickness=None,
                    intersectionOpacity=None,
                    intersectionVisibility=False):
    # source: https://www.vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/OrientedArrow
    # create vtk arrow model
    arrow = vtk.vtkArrowSource()
    arrow.SetShaftRadius(0.05)
    arrow.SetShaftResolution(15)
    arrow.SetTipLength(0.3)
    arrow.SetTipResolution(15)
    arrow.Update()
    #arrow.SetInputData()
    #arrow.Update()
    # Transform the arrow's polydata
    #transformPD = vtk.vtkTransformPolyDataFilter()
    #transformPD.SetTransform(vtkTrf)
    #transformPD.SetInputConnection(arrow.GetOutputPort())
    #transformPD.Update()
    # Create model node and add to scene
    modelsLogic = slicer.modules.models.logic()
    model = modelsLogic.AddModel(arrow.GetOutput())
    if name is None:
        name = 'arrow'
    model.SetName(name)
    # change visualization
    modelChangeVisualization(name,color=color,
                    opacity=opacity,
                    visibility=visibility,
                    intersectionOpacity=intersectionOpacity,
                    intersectionVisibility=intersectionVisibility)
    # transform the arrow
    # calculate 4x4 transformation matrix from points start, end
    # vector between two points
    vx = np.array(end)-np.array(start)
    vlength = np.linalg.norm(vx)
    # vx becomes the new x-axis of a transform/target coordinate system (which is otherwise random in dy and dz)
    dx = vx/vlength
    # random axes dy and dz
    vyrand = np.random.rand(3,)
    vy = np.cross(vx,vyrand)
    dy = vy/np.linalg.norm(vy)
    vz = np.cross(dx,dy)
    dz = vz/np.linalg.norm(vz)
    # rotation matrix composed of three orthonormal columns [dx,dy,dz]
    R = np.array([dx,dy,dz]).T
    # translation is the start point
    t = np.array(start)
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    print(T)
    # scale transform with the length of the vector
    T_scale = np.eye(4)
    np.fill_diagonal(T_scale,[vlength,vlength,vlength,1.0])
    T = np.dot(T,T_scale)
    print(T)
    #M = transform4x4FromNormalAndPoint(normal,start)
    nodeTrfArrow = transformFrom4x4Matrix('trf_'+name, T)
    model.SetAndObserveTransformNodeID(nodeTrfArrow.GetID())

# draw a little coordinate system
def drawCoordinateSystem(dx=[1.0,0.0,0.0],
                         dy=[0.0,1.0,0.0],
                         dz=[0.0,0.0,1.0],
                         offset=[0.0,0.0,0.0],
                         scale=20.0,
                         colors=['Red','Green','Blue'],
                         model_prefix='coord_sys_',
                         visibility=[True,True,True],
                         createModel=[True,True,True]):     
    for dc,dir_id,color,flagvis,flagcreate in zip([dx,dy,dz],['dx','dy','dz'],colors,visibility,createModel):
        dc = np.array(dc)
        start = np.array(offset)
        end = offset + scale*dc
        if flagcreate:
            if isinstance(color, str):
                col = getColorRGBByName(color)
            else:
                col = color
            drawArrow(start,end,name=model_prefix+dir_id,color=col,visibility=flagvis)


def runModelMaking(volLabelsNode,
                   modelHierarchyName=None,
                   nameTransform=None,
                   paramSmooth=15,
                   paramDecimate=0.15,
                   paramGenerateAll=True,
                   paramJointSmoothing=True,
                   paramName='region',
                   paramModelOpacity=None,
                   paramModelColor=None,
                   paramModelSliceIntersection=None):
    # retrieve or create model hierarchy by name
    if modelHierarchyName is None:
        modelHierarchyName = 'models'
    nodeModelHierarchy = getNode(modelHierarchyName)
    if nodeModelHierarchy is None:
        nodeModelHierarchy = slicer.vtkMRMLModelHierarchyNode()
        nodeModelHierarchy.SetName(modelHierarchyName)
        slicer.mrmlScene.AddNode( nodeModelHierarchy )

    # run model maker
    parameters = {}
    parameters["InputVolume"] = volLabelsNode.GetID()
    parameters["ModelSceneFile"] = nodeModelHierarchy.GetID()
    parameters["GenerateAll"] = paramGenerateAll
    parameters["Name"] = paramName
    parameters["Smooth"] = paramSmooth
    parameters["Decimate"] = paramDecimate
    parameters["JointSmoothing"] = paramJointSmoothing
    mm = slicer.modules.modelmaker
    success = (slicer.cli.run(mm, None, parameters,wait_for_completion=True))

    # if a transform exists:
    # for each model in hierarchy, set transform to outputTransformLinear
    if nameTransform is not None:
        nodetrf = getOrCreateTransformNode(nameTransform)
    else:
        nodetrf = None

    nrNodes = nodeModelHierarchy.GetNumberOfChildrenNodes()
    #print('Number of nodes in model Hierarchy: %d' % nrNodes)
    for n in range(nrNodes):
        subHierarchy = nodeModelHierarchy.GetNthChildNode(n)
        model = subHierarchy.GetModelNode()
        if nodetrf is not None:
            model.SetAndObserveTransformNodeID(nodetrf.GetID())
        if paramModelOpacity is not None:
            model.GetDisplayNode().SetOpacity(paramModelOpacity)
        if paramModelColor is not None:
            model.GetDisplayNode().SetColor(paramModelColor)
        if paramModelSliceIntersection is not None:
            model.GetDisplayNode().SliceIntersectionVisibility(paramModelSliceIntersection)
        #print('Model %d transformed' % n)
        model.Modified()


def runGrayscaleModelMaker(volname, modelname, threshold,
                           nameTransform=None,
                           paramSmooth=15,
                           paramDecimate=0.15,
                           paramGenerateAll=True,
                           paramJointSmoothing=True,
                           paramSplitNormals=True,
                           paramPointNormals=True,
                           paramModelOpacity=None,
                           paramModelSliceIntersection=None):
    volnode = getNode(volname)
    if volnode is None:
        print('Volume node with name "%s" does not exist'%volname)
        return
    model = getOrCreateModelNode(modelname)

    # run model maker
    parameters = {}
    parameters["InputVolume"] = volnode.GetID()
    parameters["OutputGeometry"] = model.GetID()
    parameters["Threshold"] = threshold
    parameters["Name"] = modelname
    parameters["Smooth"] = paramSmooth
    parameters["SplitNormals"] = paramSplitNormals
    parameters["PointNormals"] = paramPointNormals
    mm = slicer.modules.grayscalemodelmaker
    success = (slicer.cli.run(mm, None, parameters,wait_for_completion=True))

    # if a transform exists:
    # for each model in hierarchy, set transform to outputTransformLinear
    if nameTransform is not None:
        nodetrf = getOrCreateTransformNode(nameTransform)
    else:
        nodetrf = None

    if nodetrf is not None:
        model.SetAndObserveTransformNodeID(nodetrf.GetID())
    if paramModelOpacity is not None:
        model.GetDisplayNode().SetOpacity(paramModelOpacity)
    if paramModelSliceIntersection is not None:
        model.GetDisplayNode().SliceIntersectionVisibility(paramModelSliceIntersection)
    model.Modified()

def modelChangeVisualization(modelname,
                             color=None,
                             opacity=None,
                             visibility=None,
                             intersectionThickness=None,
                             intersectionOpacity=None,
                             intersectionVisibility=None,
                             faceCulling=None): 
    # visibility ... has to be a boolean True/False
    # color ... has to be a tuple of (float,float,float)
    # opacity ... has to be a float in range [0...1]
    # intersectionVisibility ... has to be a boolean True/False
    # intersectionThickness ... has to be an int (thickness in voxels)
    # intersectionOpacity ... has to be a float in range [0...1]
    # faceCulling ... if not None, sets visible sides, to either 'front', 'back', or 'all'
    model = getOrCreateModelNode(modelname)
    if color is not None:
        model.GetDisplayNode().SetColor(color)
    if opacity is not None:
        model.GetDisplayNode().SetOpacity(opacity)
    if visibility is not None:
        model.GetDisplayNode().SetVisibility(visibility)
    if intersectionThickness is not None:
        model.GetDisplayNode().SetSliceIntersectionThickness(intersectionThickness)
    if intersectionOpacity is not None:
        model.GetDisplayNode().SetSliceIntersectionOpacity(intersectionOpacity)
    if intersectionVisibility is not None:
        model.GetDisplayNode().SetSliceIntersectionVisibility(intersectionVisibility)
    if faceCulling is not None:
        # TODO: this does not really work... the binary flags are neither OR, AND, nor XOR... there seems to be sth wrong on Slicer side.
        if faceCulling=='front':
            model.GetDisplayNode().SetFrontfaceCulling(1)
            model.GetDisplayNode().SetBackfaceCulling(0)
        if faceCulling=='back':
            model.GetDisplayNode().SetFrontfaceCulling(0)
            model.GetDisplayNode().SetBackfaceCulling(1)
        if faceCulling=='all':
            model.GetDisplayNode().SetFrontfaceCulling(2)
            #model.GetDisplayNode().SetBackfaceCulling(2)
        model.GetDisplayNode().Modified()

'''
OTHER UTILS
'''

def getDistinctColors(n=None):
    # colors from Sasha Trubetskoy
    # https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    
    if n is None:
        n = 22
    
    colname = []
    colhex = []
    colrgb = []
    colcmyk = []
    colname.append('Red');     colhex.append(0xe6194b); colrgb.append(tuple(x/255.0 for x in (230, 25, 75)));   colcmyk.append((0, 100, 66, 0))
    colname.append('Green');   colhex.append(0x3cb44b); colrgb.append(tuple(x/255.0 for x in (60, 180, 75)));   colcmyk.append((75, 0, 100, 0))
    colname.append('Yellow');  colhex.append(0xffe119); colrgb.append(tuple(x/255.0 for x in (255, 225, 25)));  colcmyk.append((0, 25, 95, 0))
    colname.append('Blue');    colhex.append(0x0082c8); colrgb.append(tuple(x/255.0 for x in (0, 130, 200)));   colcmyk.append((100, 35, 0, 0))
    colname.append('Orange');  colhex.append(0xf58231); colrgb.append(tuple(x/255.0 for x in (245, 130, 48)));  colcmyk.append((0, 60, 92, 0))
    colname.append('Purple');  colhex.append(0x911eb4); colrgb.append(tuple(x/255.0 for x in (145, 30, 180)));  colcmyk.append((35, 70, 0, 0))
    colname.append('Cyan');    colhex.append(0x46f0f0); colrgb.append(tuple(x/255.0 for x in (70, 240, 240)));  colcmyk.append((70, 0, 0, 0))
    colname.append('Magenta'); colhex.append(0xf032e6); colrgb.append(tuple(x/255.0 for x in (240, 50, 230)));  colcmyk.append((0, 100, 0, 0))
    colname.append('Lime');    colhex.append(0xd2f53c); colrgb.append(tuple(x/255.0 for x in (210, 245, 60)));  colcmyk.append((35, 0, 100, 0))
    colname.append('Pink');    colhex.append(0xfabebe); colrgb.append(tuple(x/255.0 for x in (250, 190, 190))); colcmyk.append((0, 30, 15, 0))
    colname.append('Teal');    colhex.append(0x008080); colrgb.append(tuple(x/255.0 for x in (0, 128, 128)));   colcmyk.append((100, 0, 0, 50))
    colname.append('Lavender');colhex.append(0xe6beff); colrgb.append(tuple(x/255.0 for x in (230, 190, 255))); colcmyk.append((10, 25, 0, 0))
    colname.append('Brown');   colhex.append(0xaa6e28); colrgb.append(tuple(x/255.0 for x in (170, 110, 40)));  colcmyk.append((0, 35, 75, 33))
    colname.append('Beige');   colhex.append(0xfffac8); colrgb.append(tuple(x/255.0 for x in (255, 250, 200))); colcmyk.append((5, 10, 30, 0))
    colname.append('Maroon');  colhex.append(0x800000); colrgb.append(tuple(x/255.0 for x in (128, 0, 0)));     colcmyk.append((0, 100, 100, 50))
    colname.append('Mint');    colhex.append(0xaaffc3); colrgb.append(tuple(x/255.0 for x in (170, 255, 195))); colcmyk.append((33, 0, 23, 0))
    colname.append('Olive');   colhex.append(0x808000); colrgb.append(tuple(x/255.0 for x in (128, 128, 0)));   colcmyk.append((0, 0, 100, 50))
    colname.append('Coral');   colhex.append(0xffd8b1); colrgb.append(tuple(x/255.0 for x in (255, 215, 180))); colcmyk.append((0, 15, 30, 0))
    colname.append('Navy');    colhex.append(0x000080); colrgb.append(tuple(x/255.0 for x in (0, 0, 128)));     colcmyk.append((100, 100, 0, 50))
    colname.append('Grey');    colhex.append(0x808080); colrgb.append(tuple(x/255.0 for x in (128, 128, 128))); colcmyk.append((0, 0, 0, 50))
    colname.append('White');   colhex.append(0xFFFFFF); colrgb.append(tuple(x/255.0 for x in (255, 255, 255))); colcmyk.append((0, 0, 0, 0))
    colname.append('Black');   colhex.append(0x000000); colrgb.append(tuple(x/255.0 for x in (0, 0, 0)));       colcmyk.append((0, 0, 0, 100))
    
    return (colrgb[0:n],colname[0:n],colhex[0:n],colcmyk[0:n])

def getColorRGBByName(name):
    if name ==  'Gray':
        name = 'Grey'
    colrgb,colname,colhex,colcmyk = getDistinctColors()
    idx = colname.index(name)
    return colrgb[idx]

# conversions of full-width-at-half-maximum to sigma and vice versa
# source: https://matthew-brett.github.io/teaching/smoothing_intro.html
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))
    
def np_prettyprint():
    print('np.set_printoptions(precision=3,suppress=True)')

# plane parameter estimation from N points
# via SVD (analoguous to PCA) - source: https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
# Subtract out the centroid, form a 3xN matrix X out of the resulting coordinates and
# calculate its singular value decomposition. The normal vector of the best-fitting plane
# is the left singular vector corresponding to the least singular value.
def lstsqPlaneEstimation(pts):
    # pts need to be of dimension (Nx3)
    # pts need to be centered before estimation of normal!!
    ptsCentered = pts-np.mean(pts,axis=0)
    # do fit via SVD
    u, s, vh = np.linalg.svd(ptsCentered[:,0:3].T, full_matrices=True)
    #print(u)
    #print(s)
    #print(vh)
    normal = u[:,-1] 
    # the normal's z-direction should point towards the world z-direction
    if normal[-1]<0:
        #normal *= -1.0
        normal[-1] *= -1.0
    # Make R orthonormal
    R = u.copy() # u is already orthonormal
    # previously:
    #R = np.zeros((3,3))
    #for i in range(3):
    #    R[:,i] = u[:,i] / np.linalg.norm(u[:,i])
    #normal = u[:,-1] / np.linalg.norm(u[:,-1])
    offset = np.mean(pts[:,0:3],axis=0)
    return (normal,offset,R)

def transform4x4FromNormalAndPoint(normal,point,pointOnPlane=None):
    # pointOnPlane should be a point that roughly indicates where "posterior" is oriented
    if pointOnPlane is None:
        pointOnPlane = np.random.rand(3,)
    # orientation vectors
    dz = normal.ravel()
    dx = np.cross(dz,pointOnPlane) # x is the cross between "up" (the new "normal"), and "front" (if pointOnPlane is pointing to the posterior) 
    dx = dx/np.linalg.norm(dx)
    if dx[0]<0:
        dx *= -1.0
    dy = np.cross(dz,dx)
    dy = dy/np.linalg.norm(dy)
    # rotation matrix
    R = np.array([dx,dy,dz]).T
    # translation part
    t = point.ravel()
    # 4x4 transform matrix
    M = np.eye(4)
    M[0:3,0:3] = R
    M[0:3,3]   = t
    return M

def transformFromPointsFitLsq(pts,
                              pointOnPlanePosterior=None,
                              nameNativeToPointNormalized='trfNativeToPointsNormalized',
                              namePointNormalizedToNative='trfPointsNormalizedToNative',
                              visualizeSVDCoordinateSystem=False,
                              visualizeCoordinateSystem=False):
    if type(pts) is str:
        pts = arrayFromFiducialList(pts)[:,0:3]
    normal,offset,R = lstsqPlaneEstimation(pts)
    midpoint = np.mean(pts,axis=0)
    if pointOnPlanePosterior is None:
        pointOnPlanePosterior = midpoint + R[:,1].ravel()
    T = transform4x4FromNormalAndPoint(normal,midpoint,pointOnPlane=pointOnPlanePosterior[0:3])
    if False:
        # calculate a new x-vector maximally aligned with the direction of the x-axis
        xnew = np.dot(R[0:3,0].ravel(),np.array([1.0,0.0,0.0]))
        xnew = xnew / np.linalg.norm(xnew)
        # calculate a new y-axis as the cross product
        ynew = np.cross(R[:,2].ravel(),xnew)
        Rnew = np.zeros((3,3))
        Rnew[:,0] = xnew
        Rnew[:,1] = ynew
        Rnew[:,2] = R[:,2]
        
        T = np.eye(4)
        T[0:3,0:3] = Rnew
        T[0:3,3] = midpoint.ravel()
    trfPointNormalizedToNative = transformFrom4x4Matrix(namePointNormalizedToNative,T)
    trfNativeToPointNormalized = transformFrom4x4Matrix(nameNativeToPointNormalized,np.linalg.inv(T))
    
def transformFromReidsPlanePoints(pts,
                                  nameNativeToReidnormalized='trfNativeToReidnormalized',
                                  nameReidnormalizedToNative='trfReidnormalizedToNative',
                                  offsetPlane='midpointMeatus',
                                  pt_idxs_eReL_mRmL_iRiL=[0,1,2,3,4,5],
                                  visualizeSVDCoordinateSystem=False,
                                  visualizeCoordinateSystem=False,
                                  noTranslation=False):
    if type(pts) is str:
        pts = arrayFromFiducialList(pts)
    idx = np.array(pt_idxs_eReL_mRmL_iRiL)
    
    # process reid points to different origins
    arr = pts[:,0:-1]
    midpointAll = np.mean(arr,axis=0).ravel()
    midpointMeatus = np.mean(arr[idx[[2,3]],:],axis=0).ravel()
    midpointOrbital = np.mean(arr[idx[[4,5]],:],axis=0).ravel()
    midpointR = np.mean(arr[idx[[2,4]],:],axis=0).ravel()
    midpointL = np.mean(arr[idx[[3,5]],:],axis=0).ravel()
    vecLR = midpointR-midpointL
    vecPA = midpointOrbital-midpointMeatus
    # plane normal --> z-axis, invert the axis if it is not pointing roughly "upwards" (normal[-1] has to be >0)
    # points to estimate normal: meatus and infraorbital points, i.e. ignoring "eye" (lateral eye lid joints)
    normal, offset, R = lstsqPlaneEstimation(arr[idx[[2,3,4,5]],:])
    if visualizeSVDCoordinateSystem:
        t = np.mean(arr[idx[[2,3,4,5]],:],axis=0).ravel()
        dx, dy, dz = R[:,0].ravel(), R[:,1].ravel(), R[:,2].ravel()
        for dc,dir_id,color in zip([dx,dy,dz],['svd_dx','svd_dy','svd_dz'],[(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]):
            start = t
            end = t + 20.0*dc
            drawArrow(start,end,name='arrow_reid_'+dir_id,color=color)
    if normal[-1]<0:
        normal = -1.0 * normal

    # create a plane origin system
    dz = normal
    # projection of PA axis onto plane --> y-axis
    vecPA_onPlane = vecPA - (np.dot(vecPA,normal))*normal
    dy = vecPA_onPlane/np.linalg.norm(vecPA_onPlane)
    # x-axis is then the cross-product of dz and dy
    dx = np.cross(dy,dz)

    # compose the transformation matrix, finally
    R = np.array([dx,dy,dz]).T
    if offsetPlane=='midpointMeatus':
        t = midpointMeatus
    elif offsetPlane=='meanReidPoints':
        t = offset
    elif offsetPlane=='rotationOnly':
        t = offset * 0.0
    else:
        print('WARNING: transformFromReidsPlanePoints: Setting offsetPlane=''%s'' is not implemented.\nDefaulting Reid''s plane origin to midpointMeatus.'%(offsetPlane))
        t = midpointMeatus
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    if noTranslation:
        T[0:3,3] = 0.0
    
    trfNativeToReidnormalized = transformFrom4x4Matrix(nameReidnormalizedToNative, T)
    trfReidnormalizedToNative = transformFrom4x4Matrix(nameNativeToReidnormalized, np.linalg.inv(T))
    
    # visualize the coordinate system
    if visualizeCoordinateSystem:        
        for dc,dir_id,color in zip([dx,dy,dz],['dx','dy','dz'],[(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]):
            start = t
            end = t + 20.0*dc
            drawArrow(start,end,name='arrow_reid_'+dir_id,color=color)

def angleBetweenTwoCoordinateTransforms():
    # source: https://math.stackexchange.com/questions/1870661/find-angle-between-two-coordinate-systems
    pass

def numpy4x4ToVtkMatrix4x4(M):
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, M[i,j])
            matrix.SetElement(i, j, M[i,j])
            matrix.SetElement(i, j, M[i,j])
    return matrix
    
def vtkMatrix4x4ToNumpy(M):
    matrix = np.eye(4)
    for i in range(4):
        for j in range(4):
            matrix[i,j] = M.GetElement(i, j)
    return matrix
    

def transform4x4FromRandT(R=None,t=None):
    T = np.eye(4)
    if R is not None:
        T[0:3,0:3] = R
    if t is not None:
        T[0:3,3] = np.array(t).ravel()
    return T
    
def transformToNumpy4x4(nameTrf):
    nTrf = getOrCreateTransformNode(nameTrf)
    T = nTrf.GetMatrixTransformFromParent()
    M = vtkMatrix4x4ToNumpy(T)
    return M

def loadJson(ffJson):
    import json
    json_file = open(ffJson)
    json_str = json_file.read()
    json_data = json.loads(json_str)
    return json_data

def computeDiceOverlap(np_arr1,np_arr2,int_foreground1=1,int_foreground2=1):
    mask1 = np_arr1==int_foreground1
    mask2 = np_arr2==int_foreground2
    intersection = np.logical_and(mask1, mask2)
    mask_sum = mask1.sum() + mask2.sum()
    if mask_sum == 0:
        dice_coeff = 1.0
    else:
        dice_coeff = 2. * intersection.sum() / (mask_sum)
    return dice_coeff

def generate_perlin_noise_3d(shape, res):
    # from: https://pvigier.github.io/2018/11/02/3d-perlin-noise-numpy.html
    # Simplex noise explained: http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
    # github repo: https://github.com/pvigier/perlin-numpy
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    gradients[-1] = gradients[0]
    g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)
    
def generate_fractal_noise_3d(shape, res, octaves=1, persistence=0.5):
    # from: https://pvigier.github.io/2018/11/02/3d-perlin-noise-numpy.html
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(shape, (frequency*res[0], frequency*res[1], frequency*res[2]))
        frequency *= 2
        amplitude *= persistence
    return noise

def np_get_otsu_threshold(arr):
    # from: https://stackoverflow.com/a/50796152
    pixel_number = np.prod(arr.shape)
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(arr.ravel(), bins=128)
    final_thresh = -np.inf
    final_value = -np.inf
    intensity_arr = bins[:-1]
    for tidx0, tval in enumerate(bins[1:-1]): 
        tidx = tidx0+1    
        pcb = np.sum(his[:tidx])
        pcf = np.sum(his[tidx:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth
        
        mub = np.sum(intensity_arr[:tidx]*his[:tidx]) / float(pcb)
        muf = np.sum(intensity_arr[tidx:]*his[tidx:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2
        
        if value > final_value:
            final_thresh = tval
            final_value = value
    return final_thresh

def np_sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def piecewise_linear_map(arr,
                         bsrc = [-1024.0, 20.0, 80.0, 2048.0], 
                         btgt = [0.0, 20.0, 150.0, 255.0]):
    
    # pre-process: clip arr intensities at bsrc min/max values
    arr = np.clip(arr, bsrc[0], bsrc[-1])
    # ranges
    rsrc = np.diff(bsrc)
    rtgt = np.diff(btgt)
    # allocate result
    res = arr.copy()
    for i in range(1,len(bsrc)):
        blower = bsrc[i-1]
        bupper = bsrc[i]
        #print('lower_src: %0.1f, upper_src: %0.1f'%(blower, bupper))
        mask = np.logical_and(arr>=blower, arr<=bupper)
        vec = arr[mask]
        # normalize to 0...1
        vec -= blower
        vec /= rsrc[i-1]
        # map to new range
        vec *= rtgt[i-1]
        vec += btgt[i-1]
        # store into array
        res[mask] = vec
    return res



def closeScene():
    slicer.mrmlScene.Clear(0)
