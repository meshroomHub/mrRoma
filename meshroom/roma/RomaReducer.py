__version__ = "1.0"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES

class RomaReducer(desc.CommandLineNode):

    category = "ROMA"
    documentation = """ RomaSampler is parallelized and distributed. It generated multiple
    output files which must merged together. """
    size = desc.DynamicNodeSize('inputSfMData')

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "reducer.py").as_posix()
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
            name="samplesFolder",
            label="Samples folder",
            description="Samples folder from RomaSampler.",
            value="",
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
