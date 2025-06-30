__version__ = "1.0"

import os 
from pathlib import Path
from meshroom.core import desc

class RomaSampler(desc.CommandLineNode):

    category = "ROMA"
    documentation = """"""
    gpu = desc.Level.INTENSIVE


    exePath = (Path(__file__).absolute().parent.parent.parent / "romaProcessor.py").as_posix()

    commandLine="python "+exePath+" sample {allParams}"

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
        )
    ]

    outputs = [
        desc.File(
            name="outputFolder",
            label="Output folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
