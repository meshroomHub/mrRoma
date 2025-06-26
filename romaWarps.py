__version__ = "2.0"

import os 

from meshroom.core import desc

class RomaWarp(desc.CommandLineNode):

    category = "ROMA"
    documentation = """"""
    gpu = desc.Level.INTENSIVE


    exePath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "roma.py")

    commandLine="rez env roma-develop -- python "+exePath+" warp {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
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
        desc.IntParam(
            name="samplingStep",
            label="samplingStep",
            description="""Sampling for Keypoint, 0 to use Roma's sampling, 1 for no sampling, otherwise will sample pixels every n pixel, on a grid""",
            value=15,
            range=(0, 20000, 1),
        ),
        desc.IntParam(
            name="keypointsPerKeyFrame",
            label="keypointsPerKeyFrame",
            description="""Number of keypoints per keyframe for roma sampling""",
            value=2000,
            range=(0, 10000000, 1),
            enabled=lambda node: node.samplingStep.value == 0,
        ),
    ]

    outputs = [
        desc.File(
            name="outputFolder",
            label="Output folder",
            description="",
            value="{nodeCacheFolder}"
        ),
        desc.File(
            name="Warps",
            label="Warps",
            description="",
            value=os.path.join("{nodeCacheFolder}", "*_warp.exr"),
            semantic="sequence",
            group=""
        ),
        desc.File(
            name="Certainty",
            label="Certainty",
            description="",
            value=os.path.join("{nodeCacheFolder}", "*_certainty.exr"),
            semantic="sequence",
            group=""
        ),
        desc.File(
            name="Sampling",
            label="Sampling",
            description="",
            value=os.path.join("{nodeCacheFolder}", "*_sampling.exr"),
            semantic="sequence",
            group=""
        ),
    ]
