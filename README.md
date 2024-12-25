# Introduction
https://blackmanba.top/doc/116/

Test on Nvidia Jetson AGX orin, Jetpack 6.1.3, l4t 36.3 tensorRT 8.6.2 cuda driver 12.2

# How to use

* Download model: https://github.com/onnx/models/blob/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx

* transform onnx 2 engine: `/usr/src/tensorrt/bin/trtexec --onnx=resnet50-v2-7.onnx --saveEngine=resnet50-v2-7.engine --useDLACore=0 --fp16 --allowGPUFallback`

* build & run:
``` bash
mkdir build && cd build
cmake ..
make

./TensorRT_Inference ../resnet50-v2-7.engine ../kitten.jpg
```
