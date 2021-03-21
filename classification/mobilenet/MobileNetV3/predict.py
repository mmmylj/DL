import os
import json
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from mobilenetv3 import mobilenet_v3_large

IMAGE_LIST = []
predict_cla_list = []

DATA_PATH = '../dataset/ILSVRC-2012/val/n01494475'
# DATA_PATH = '/home/jieliu/workspace/vela_model/dataset/ILSVRC-2012/val/n01440764'
WEIGHT_PATH = "./weight"
LABEL_PATH = "/home/jieliu/workspace/vela_model/dataset/ILSVRC-2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"

MOBILENET_LARGE_WEIGHT = os.path.join(
    WEIGHT_PATH, 'mobilenet_v3_large-8738ca79.pth')
MOBILENET_SMALL_WEIGHT = os.path.join(
    WEIGHT_PATH, 'mobilenet_v3_small-047dcff4.pth')


def main():
    true_num = 0
    with open(LABEL_PATH, "r") as f:
        label_list = f.readlines()
    print(label_list[11233])

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if (file.split('.')[-1] == 'JPEG'):
                filename = os.path.join(root, file)
                IMAGE_LIST.append(filename)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transform_gray = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # create model
    model = mobilenet_v3_large().to(device)
    # load model weights
    model_weight_path = MOBILENET_LARGE_WEIGHT
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    for IMAGE_PATH in IMAGE_LIST:
        # load image
        assert os.path.exists(
            IMAGE_PATH), "file: '{}' dose not exist.".format(IMAGE_PATH)
        img = Image.open(IMAGE_PATH)
        plt.imshow(img)
        # [N, C, H, W]
        print(IMAGE_PATH)
        IMAGE_ID = int(IMAGE_PATH.split('.')[-2].split('_')[-1])
        if len(np.shape(img)) == 2:
            img = data_transform_gray(img)
        elif np.shape(img)[-1] == 4:
            img = img.convert("RGB")
            img = data_transform(img)
        else:
            img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # predict class

            # outputs = model(img)
            # _, predicted=torch.max(outputs.data, 1)
            # print("pre train value: {}".format(predicted))

            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            predict_cla_list.append(predict_cla)
            print("pre train value: {}".format(predict_cla))
            print("label: {}".format(int(label_list[IMAGE_ID - 1])))
            # if predict_cla == int(label_list[IMAGE_ID - 1]):
            #     true_num += 1

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        # print(print_res)
        # plt.show()
    # print("acc is {}".format(true_num / len(predict_cla_list)))
if __name__ == '__main__':
    main()
