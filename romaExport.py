__version__ = "2.0"

import os 

from meshroom.core import desc

class RomaExport(desc.CommandLineNode):

    category = "ROMA"
    documentation = """"""
    gpu = desc.Level.INTENSIVE


    exePath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "roma.py")

    commandLine="rez env roma-develop -- python "+exePath+" export {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="wrapFolder",
            label="wrapFolder",
            description="Input wrap folder.",
            value="",
        ),
        desc.File(
            name="keyFrameSfMData",
            label="keyFrameSfMData",
            description="Input Keyframe SfMData (will overwite sampling).",
            value="",
        ),
        desc.IntParam(
            name="keyframeSteps",
            label="keyframeSteps",
            description="""Steps to set each keyframe""",
            value=20,
            range=(0, 20000, 1),
            advanced=False,
            enabled=lambda node: node.keyFrameSfMData.value == "",
        ),
        
        desc.File(
            name="maskFolder",
            label="maskFolder",
            description="Input mask folder.",
            value="",
        ),
 
        desc.FloatParam(
            name="confThreshold",
            label="confThreshold",
            description="""Confidence threshold with witch to automatically reject matches""",
            value=0.05,
            range=(0.0, 1.0, 0.01),
            advanced=True,
        ),
    ]

    outputs = [
        desc.File(
            name="outputFolder",
            label="Output folder",
            description="",
            value="{nodeCacheFolder}"
        ),
    ]
