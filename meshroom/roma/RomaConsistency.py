__version__ = "1.0"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar


class RomaConsistency(desc.CommandLineNode):

    category = "ROMA"
    documentation = """
    Update confidence map based on consistency.
    The consistency is checked per triplet.
    A triplet is composed of one item (A) in the reference SfmData and two consecutive items (B,C) in the framesSfmData.
    framesSfmData is assumed to be a sequence.
    We check that the estimated cumulated motion A->B->C is close to the motion of A->C
    """    

    size = avpar.DynamicViewsSize('referenceSfMData')
    parallelization = desc.Parallelization(blockSize=5)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"
    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "consistency.py").as_posix()
    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="referenceSfMData",
            label="Reference SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="framesSfMData",
            label="Frames SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.ListAttribute(
            name="warpFolders",
            label="Warp folders",
            description="List of folders where warp files can be looked up.",
            elementDesc=desc.File(
                name="warpFolder",
                label="Warp folder",
                description="",
                value=""
            ),
            exposed=True
        ),
        desc.ListAttribute(
            name="confidenceFolders",
            label="Confidence folders",
            description="List of folders where confidence files can be looked up.",
            elementDesc=desc.File(
                name="confidenceFolder",
                label="Confidence folder",
                description="",
                value=""
            ),
            exposed=True
        ),
        desc.FloatParam(
            name="maxDistance",
            label="Maximal distance for consistency",
            description="Minimal confidence threshold in pixels (in the size of the roma warp image).",
            value=32.0,
            range=(0.0, 64.0, 1.0)
        )
    ]

    outputs = [
        desc.File(
            name="outputConfidenceFolder",
            label="Output Confidence folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
