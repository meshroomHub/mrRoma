__version__ = "1.0"

import os 
from pathlib import Path

from meshroom.core import desc

class RomaMatcher(desc.CommandLineNode):

    category = "ROMA"
    documentation = """
    Compute ROMA warp and certainty images on a list of images pairs.
    """
    size = desc.DynamicNodeSize('inputSfMData')
    gpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=40)
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
            name="outputCertaintyFolder",
            label="Output Certainty folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
