import torchvision
import torchvision.models as models
import torch
import torchvision.transforms as transforms
from label import classes

DATAPATH = "/home/jieliu/workspace/vela_model/MobileNetV3/dataset/ILSVRC-2012"


#pytorch officcal model & pretrain weight

mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_large.half()
mobilenet_v3_large.eval()


transform = transforms.Compose(
    [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testset = torchvision.datasets.ImageNet(
    DATAPATH, "val", download = False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

dataiter = iter(testloader)
images, labels = dataiter.next()

correct = 0
total = 0

with torch.no_grad():
    i = 0
    for data in testloader:
        images, labels = data
        images = images.half()
        outputs = mobilenet_v3_large(images)
        _, predicted=torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the epoch: %d test images: %d %%' % (i,
            100 * correct / total))
        i += 1
        if i == 10:
            break

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
