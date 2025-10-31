# Roma Nodes

MatchMasking : Apply reference and matched images' masks to the uncertainty images.

Roma Matcher : Takes a sfmData + a list of pairs to compute and output a warp image and a certainty image per pair.

RomaSampler : Output features and matches given the output of Roma Matcher

RomaReducer : Reduce the variout parallelized outputs from RomaSampler into a single usable output

StarListing : Create a list of pairs which are connecting keyframes to the neighboor frames.