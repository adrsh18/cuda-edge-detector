# Canny Edge Detector on CUDA
Detects edges in a given image or picture using Canny Edge Detection algorithm.

#### Build
To build both serial and parallel (CUDA) binaries:
```
make all
```
For parallel (CUDA) build:
```
make
#or
make parallel
```
For serial build:
```
make serial
```

#### Run Binary
```
./edge_detector <input_image_name> <output_image_name> <high_threshold> <low_threshold>
```
Example:
```
./edge_detector eval_images/porsche4.png eval_images/out_porsche4.png 100 50
```
