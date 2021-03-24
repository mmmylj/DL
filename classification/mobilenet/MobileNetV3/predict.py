import torchvision
import torchvision.models as models
import torch
import torchvision.transforms as transforms
from label import classes
import torch.onnx

import mobilenetv3
import mobilenetv3_to_onnx

#ImageNet dataset path
# DATAPATH = "/home/jieliu/workspace/vela_model/MobileNetV3/dataset/ILSVRC-2012"
DATAPATH = "/home/jieliu/workspace/vela_model/dataset/ILSVRC-2012"

# /software/topsinference_qa/models/topsinference_test_mobilenet_v3_large_fp32.onnx
mobilenet_v3_large = mobilenetv3.mobilenet_v3_large(pretrained=True)
# mobilenet_v3_large.half()
mobilenet_v3_large.eval()

def pytorch_to_onnx(model, input_shape, model_name):

    # Input to the model
    x = torch.randn(input_shape, requires_grad=True)
    torch_out = mobilenet_v3_large(x)

    # Export the model
    torch.onnx.export(  model,                                  # model being run
                        x,                                      # model input (or a tuple for multiple inputs)
                        model_name,                             # where to save the model (can be a file or file-like object)
                        export_params=True,                     # store the trained parameter weights inside the model file
                        opset_version=9,                        # the ONNX version to export the model to
                        do_constant_folding=True,               # whether to execute constant folding for optimization
                        input_names = ['input'],                # the model's input names
                        output_names = ['output'],              # the model's output names
                        dynamic_axes={'input' : {0 : '-1'},     # variable lenght axes
                                      'output' : {0 : '-1'}})

def predict(model, datapath, us_fp16=False):
    transform = transforms.Compose(
        [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    testset = torchvision.datasets.ImageNet(
        datapath, "val", download = False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    correct = 0
    total = 0

    with torch.no_grad():
        i = 0
        for data in testloader:
            images, labels = data

            if us_fp16:
                images = images.half()

            outputs = model(images)
            _, predicted=torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the epoch: %d test images: %d %%' % (i,
                100 * correct / total))

            i += 1
    print('Accuracy of the network on the 50000 test images: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':

    # ImageNet dataset path
    # DATAPATH = "/home/jieliu/workspace/vela_model/MobileNetV3/dataset/ILSVRC-2012"
    DATAPATH = "/home/jieliu/workspace/vela_model/dataset/ILSVRC-2012"

    # The official implementation of Pytorch for the mobilenet_v3_large (torchvision >= 0.9.0)
    # mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)

    # The official implementation of Pytorch for the mobilenet_v3_large (torchvision < 0.9.0)
    # mobilenet_v3_large = mobilenetv3.mobilenet_v3_large(pretrained=True)

    mobilenet_v3_large = mobilenetv3_to_onnx.mobilenet_v3_large(
        pretrained=True)
    # mobilenet_v3_large.half()
    mobilenet_v3_large.eval()
    # pytorch_to_onnx(mobilenet_v3_large, [1, 3, 224, 224], "mobilenet_v3_large_fp32.onnx")
    predict(mobilenet_v3_large, DATAPATH, False)
