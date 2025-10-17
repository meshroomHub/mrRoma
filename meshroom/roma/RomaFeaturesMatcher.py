__version__ = "1.0"

import os 
from pathlib import Path

from meshroom.core import desc

class RomaFeaturesMatcher(desc.CommandLineNode):

    category = "ROMA"
    documentation = """"""
    size = desc.DynamicNodeSize('inputSfMData')
    gpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=40)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"
    

    exePath = (Path(__file__).absolute().parent.parent.parent / "featuresMatcher.py").as_posix()

    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="featuresFolder",
            label="Features folder",
            description="Input features",
            value=""
        ),
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Path to a file which contains the list of image pairs to match.",
            value="",
        ),
        desc.File(
            name="warpFolder",
            label="warp folder",
            description="",
            value=""
        ),
        desc.File(
            name="masksFolder",
            label="Masks folder",
            description="",
            value=""
        ),
        desc.ChoiceParam(
            name="masksExtension",
            label="Mask File Extension",
            description="Mask file extension",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
        ),
        desc.FloatParam(
            name="minCertainty",
            label="Minimal certainty",
            description="Minimal certainty threshold.",
            value=0.15,
            range=(0.0, 1.0, 0.01)
        )
    ]

    outputs = [
        desc.File(
            name="matchesFolder",
            label="Output folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
