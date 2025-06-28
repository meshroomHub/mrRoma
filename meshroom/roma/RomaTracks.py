__version__ = "2.0"

import os
from pathlib import Path

from meshroom.core import desc

class RomaTracksCreation(desc.CommandLineNode):

    category = "ROMA"
    documentation = """"""
    gpu = desc.Level.INTENSIVE

    exePath = (Path(__file__).absolute().parent.parent.parent / "roma.py").as_posix()

    commandLine="python "+exePath+" tracks {allParams}"

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
        desc.IntParam(
            name="minTrackLength",
            label="minTrackLength",
            description="""Remove tracks shorter than the given length""",
            value=2,
            range=(2, 20000, 1),
            advanced=True,
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
            name="outputTracks",
            label="Output tracks",
            description="",
            value=os.path.join("{nodeCacheFolder}", "tracks.json")
        )
    ]
