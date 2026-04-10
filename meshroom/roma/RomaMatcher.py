__version__ = "1.1"

import os 
from pathlib import Path
from meshroom.core import desc
from pyalicevision import parallelization as avpar

class RomaMatcher(desc.CommandLineNode):

    category = "ROMA"
    documentation = """
    Compute ROMA warp and confidence images on a list of images pairs.
    """
    size = avpar.DynamicViewsSize('inputSfMData')
    gpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=20)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"
    

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "matcher.py").as_posix()

    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Path to a file which contains the list of image pairs to match.",
            value="",
        ),
        desc.BoolParam(
            name="checkLoops",
            label="Check loop consitency",
            description="Check that there is a consistency between A-B and B-A.",
            value=False
        ),
        desc.FloatParam(
            name="loopThreshold",
            label="Allowed loop error",
            description="Loop consistency max error. The distance between the original coordinates and the back and forth coordinates.",
            value=3.0,
            range=(0.0, 10.0, 1.0),
            enabled=lambda node: node.checkLoops.value
        ),
        desc.BoolParam(
            name="outputCovarianceFlag",
            label="Output covariance",
            description="Output covariance image",
            value=False
        ),
        desc.FloatParam(
            name="minConfidence",
            label="Minimal Confidence",
            description="Minimal confidence threshold under which the output will be zeroed for compression reasons.",
            value=0.02,
            range=(0.0, 1.0, 0.01)
        )
    ]

    outputs = [
        desc.File(
            name="outputWarpFolder",
            label="Output Warp folder",
            description="",
            value="{nodeCacheFolder}"
        ),
        desc.File(
            name="outputConfidenceFolder",
            label="Output Confidence folder",
            description="",
            value="{nodeCacheFolder}"
        ),
        desc.File(
            name="outputCovarianceFolder",
            label="Output Covariance folder",
            description="",
            value="{nodeCacheFolder}",
            enabled=lambda node : node.outputCovarianceFlag.value
        )
    ]
