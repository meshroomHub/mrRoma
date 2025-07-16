__version__ = "1.0"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES

class RomaSampler(desc.CommandLineNode):

    category = "ROMA"
    documentation = """"""
    size = desc.DynamicNodeSize('inputSfMData')
    gpu = desc.Level.INTENSIVE
    
    exePath = (Path(__file__).absolute().parent.parent.parent / "sampler.py").as_posix()
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
        desc.IntParam(
            name="maxMatches",
            label="Requested matches",
            description="All Uncertainties from all pairs starting from the same view are mixed to enforce connections.",
            value=10000,
            range=(0, 50000, 1000)
        ),
        desc.FloatParam(
            name="minCertainty",
            label="Minimal certainty",
            description="Minimal certainty threshold.",
            value=0.15,
            range=(0.0, 1.0, 0.01)
        ),
        desc.File(
            name="filtersFolder",
            label="Filters Folder",
            description="Json files containing the estimated geometric filters",
            value=""
        ),
        desc.BoolParam(
            name="groupUncertainties",
            label="Group uncertainties",
            description="All Uncertainties from all pairs starting from the same view are mixed to enforce connections.",
             value=False
        ),
        desc.ChoiceParam(
            name="describerTypes",
            label="Describer Types",
            description="Describer types used to describe an image.",
            values=DESCRIBER_TYPES,
            value=["sift"],
            exclusive=False,
            joinChar=",",
            group="ingored"
        ),
    ]

    outputs = [
        desc.File(
            name="featuresFolder",
            label="Output Features folder",
            description="",
            value="{nodeCacheFolder}"
        ),

        desc.File(
            name="matchesFolder",
            label="Output Matches folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
