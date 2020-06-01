import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import time
import pandas as pd


def extract_one_image(input_batch, layer_index):
    l = model.features[layer_index].expand1x1_activation # need to modify
    layer = []
    def hook(module, input, output):
        layer.append(output.clone().detach())

    handle = l.register_forward_hook(hook)

    with torch.no_grad():
        model(input_batch).to(device)
    handle.remove()
    return layer[0].numpy()[0].flatten()


def create_memmap(file, ncols):
    # filename = '/mnt/xuyuhang/pytorch-cnn-visualizations/dataset_10_1000/' + file
    filename = '/mnt/xuyuhang/guided_backprop/dataset/' + file
    return np.memmap(filename, dtype=np.float32, mode='w+', shape=(nsamples, ncols))


if __name__ == "__main__":
    device = torch.device("cuda:4")
    os.environ['TORCH_HOME'] = '/mnt/xuyuhang/EDGE'
    model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=True)
    time_start = time.time()
    SOURCE = '/mnt/xuyuhang/train/'
    nsamples = 1000
    # dataset_input = create_memmap('n03759954.dat', 150528)
    df_GT = pd.read_csv('/mnt/xuyuhang/EDGE/Combined_mapping_squeeze.csv')
    array_GT = np.array(df_GT)
    dict_GT = dict(zip(array_GT[:, 3], array_GT[:, 2]))
    count = 0
    # for j, img in enumerate(os.listdir(SOURCE + 'n03759954/')[:1000]):
    #     if j % 100 == 0:
    #         print('.', end="")
    #     filename = SOURCE + 'n03759954/' + img
    #     input_image = Image.open(filename)
    #     if input_image.getbands() != ('R', 'G', 'B'):
    #         continue
    #     preprocess = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     input_tensor = preprocess(input_image)
    #     input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    #     dataset_input[count, ] = input_batch.numpy()[0].flatten()
    #     count += 1
    # print(count)

    # expand3x3_activation: 186624, 186624, 373248, 93312, 139968, 139968, 186624, 43264
    # squeeze_activation: 46656, 46656, 93312, 23328, 34992, 34992, 46656, 10816
    # maxpooling experiment: 774400, 186624, 373248, 86528
    # concat: 373248, 373248, 746496, 186624, 279936, 279936, 373248, 86528

    dataset_fire1 = create_memmap('expand1x1_activation_fire1.dat', 186624)
    # dataset_fire2 = create_memmap('expand1x1_activation_fire2.dat', 186624)
    # dataset_fire3 = create_memmap('expand1x1_activation_fire3.dat', 373248)
    # dataset_fire4 = create_memmap('expand1x1_activation_fire4.dat', 93312)
    # dataset_fire5 = create_memmap('expand1x1_activation_fire5.dat', 139968)
    # dataset_fire6 = create_memmap('expand1x1_activation_fire6.dat', 139968)
    # dataset_fire7 = create_memmap('expand1x1_activation_fire7.dat', 186624)
    # dataset_fire8 = create_memmap('expand1x1_activation_fire8.dat', 43264)

    for i, dir in enumerate(os.listdir(SOURCE)[:10]):
        print(i, dict_GT[dir], count)
        subfolder = SOURCE + dir + '/'
        for j, img in enumerate(os.listdir(subfolder)[:1000]):
            if j % 100 == 0:
                print('.', end="")
            filename = subfolder + img
            input_image = Image.open(filename)
            if input_image.getbands() != ('R', 'G', 'B'):
                continue
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

            flatten_img_1 = extract_one_image(input_batch, 3) # 3
            # flatten_img_2 = extract_one_image(input_batch, 4) # 4
            # flatten_img_3 = extract_one_image(input_batch, 5)
            # flatten_img_4 = extract_one_image(input_batch, 7)
            # flatten_img_5 = extract_one_image(input_batch, 8)
            # flatten_img_6 = extract_one_image(input_batch, 9)
            # flatten_img_7 = extract_one_image(input_batch, 10)
            # flatten_img_8 = extract_one_image(input_batch, 12)
            # dataset_fire7[count,] = flatten_img_7
            # flatten_img_1 = extract_one_image(input_batch, 2)  # maxpooling1
            # flatten_img_2 = extract_one_image(input_batch, 6)  # maxpooling4
            # flatten_img_3 = extract_one_image(input_batch, 11)  # maxpooling8

            # dataset_input[count, ] = input_batch.numpy()[0].flatten()
            dataset_fire1[count, ] = flatten_img_1
            # dataset_fire2[count, ] = flatten_img_2
            # dataset_fire3[count, ] = flatten_img_3
            # dataset_fire3[count, ] = flatten_img_3
            # dataset_fire4[count, ] = flatten_img_4
            # dataset_fire5[count, ] = flatten_img_5
            # dataset_fire6[count, ] = flatten_img_6
            # dataset_fire7[count, ] = flatten_img_7
            # dataset_fire8[count, ] = flatten_img_8
            # dataset_label[count, ] = dict_GT[dir]
            count += 1
    print(count)
