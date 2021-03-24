这个demo主要rereash moblinet 系列网络


测试环境
--------
1. ImageNet2012 官方数据集
2. python3.7
3. pytorch


进展
-----
基于Pytorch官方实现指定混精方案

官方方实现:
https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py


主要文件
---------
mobilenet.py : 官方实现

mobilenetv2.py : 官方实现

mobilenetv3.py : 官方实现

mobilenetv3_to_onnx.py : 为实现onnx转换对一些算子进行了重写

predict.py : 执行入口

