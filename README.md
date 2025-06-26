# Roma Nodes

Various node to run roma and process the results. It uses the corresponding "roma" environement. 
Buy default use a "star" logic eg for a set of images I_n, will select keyframes of index m, and match I_m with I_{n+i}, i the size of a sliding windows.

The logic can be found in roma.py, a command line with 3 modes:
* warp: will export the warp image, confidence and sampling maps
* track: will create tracks from warp folder
* export: will convert warp images into feature and match files
