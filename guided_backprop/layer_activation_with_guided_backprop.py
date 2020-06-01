"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU
import numpy as np
import heapq
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from misc_functions import (get_example_params, save_gradient_images, preprocess_image)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in list(self.model.features.named_modules())[1:]:
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, cnn_layer, filter_pos, layer_name):
        self.model.zero_grad()
        # l = self.model.features[cnn_layer]  # need to modify
        # layer = []
        # def hook(module, input, output):
        #     layer.append(output.clone())
        # handle = l.register_forward_hook(hook)
        # with torch.set_grad_enabled(True):
        #     self.model(input_image)
        # # handle.remove()
        # conv_output = torch.sum(torch.abs(layer[0][0, filter_pos]))

        # Forward pass
        x = input_image
        for index, layer in enumerate(self.model.features):
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                if layer_name == 'concat':
                    x = layer(x)
                    break
                elif layer_name == 'squeeze':
                    x = layer.squeeze(x)
                    x = layer.squeeze_activation(x)
                    break
                elif layer_name == 'expand1x1':
                    x = layer.squeeze(x)
                    x = layer.squeeze_activation(x)
                    x = layer.expand1x1(x)
                    x = layer.expand1x1_activation(x)
                    break
                elif layer_name == 'expand3x3':
                    x = layer.squeeze(x)
                    x = layer.squeeze_activation(x)
                    x = layer.expand3x3(x)
                    x = layer.expand3x3_activation(x)
                    break

            if index in [3, 4, 5, 7, 8, 9, 10, 12]:
                x = layer.squeeze_activation(layer.squeeze(x))
                x = torch.cat([
                    layer.expand1x1_activation(layer.expand1x1(x)),
                    layer.expand3x3_activation(layer.expand3x3(x))
                ], 1)
            else:
                x = layer(x)
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))

        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def find_topx_activation(dataset_path, dataset_dim, weight_dim):
    img_SOURCE = '/mnt/xuyuhang/train'
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, dataset_dim))
    if np.isnan(np.sum(dataset)) == 1:
        print('spatial: nan')
        dataset = np.nan_to_num(dataset)
    # choose one category: n02096437, 194, dandie_dinmount
    # n02930766, 468, cab, 4957:5944
    # n03759954: 650
    dataset = dataset[4957:5945, :]

    activation_sum = []
    activation_dim = dataset_dim // weight_dim
    print('activation_dim: ', activation_dim)
    for i in range(weight_dim):
        activation_sum.append(np.sum(dataset[:, activation_dim * i:activation_dim * (i + 1)], 1))
    activation_sum = np.array(activation_sum)
    print('activation_sum shape: ', activation_sum.shape)

    img_topx_value = []
    img_topx_index = []
    for filter_pos in range(weight_dim):
        filter_activation_sum = activation_sum[filter_pos, :].tolist()
        topx_value = heapq.nlargest(9, filter_activation_sum)
        img_topx_value.append(topx_value)

        topx_index = list(map(filter_activation_sum.index, topx_value))
        img_topx_index.append(topx_index)
    img_topx_index = np.array(img_topx_index)  # correspond to the filename index
    img_topx_value = np.array(img_topx_value)
    print('img_topx_index shape:', img_topx_index.shape)
    print('img_topx_value shape:', img_topx_value.shape)

    filter_topx_activation_sum = list(np.sum(img_topx_value, 1)) # how to choose important filters, need to modify.
    filter_topx_index = list(map(filter_topx_activation_sum.index, heapq.nlargest(10, filter_topx_activation_sum)))
    img_name_list= []
    with open('img_name_list.txt', 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            img_name_list.append(img_SOURCE + currentPlace)
    img_name_array = np.array(img_name_list)[4957:5945]
    print('num of img: ', len(img_name_list))
    topx_img_name_array = img_name_array[img_topx_index]
    topx_img_name_array_part = topx_img_name_array[filter_topx_index, :]
    print('selected img shape: ', topx_img_name_array_part.shape)
    return topx_img_name_array_part.flatten(), filter_topx_index


def find_GT_for_CE(dataset_path):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, 1))
    dataset = np.nan_to_num(dataset)
    ground_truth_list = [284, 979, 511, 628, 320, 468, 460, 547, 194, 809]
    for index, gt in enumerate(ground_truth_list):
        dataset[dataset==gt] = index
    return dataset


def preprocess_images_original(img_path_list):
    prep_img_list = []
    for index, img_path in enumerate(img_path_list):
        input_image = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        preprocess_img = preprocess(input_image)
        prep_img_list.append(preprocess_img)
    return prep_img_list


def preprocess_image_model(img_path_list):
    prep_img_list = []
    for index, img_path in enumerate(img_path_list):
        input_image = Image.open(img_path).convert('RGB')
        prep_img = preprocess_image(input_image)
        prep_img_list.append(prep_img)
    return prep_img_list


def gbp_visualization(pretrained_model, prep_img_list, cnn_layer, filter_pos, layer_name):
    GBP = GuidedBackprop(pretrained_model)
    gbp_img_list = []
    for prep_img in prep_img_list:
        guided_grads = GBP.generate_gradients(prep_img, cnn_layer, filter_pos, layer_name)

        # normalize
        guided_grads = (guided_grads - np.mean(guided_grads)) / (np.std(guided_grads) + 1e-8) * 0.1
        guided_grads = np.clip(guided_grads + 0.5, 0, 1)

        # standardize
        guided_grads = guided_grads - guided_grads.min()
        guided_grads /= guided_grads.max()

        guided_grads = guided_grads.transpose(1, 2, 0)
        guided_grads = (guided_grads * 255).astype(np.uint8)
        guided_grads = Image.fromarray(guided_grads)
        gbp_img_list.append(guided_grads)
    return gbp_img_list


def save_images(img_list, filename):
    IMAGE_SIZE = 224
    IMAGE_ROW = 10
    IMAGE_COLUMN = 9
    IMAGE_SAVE_PATH = '/mnt/xuyuhang/guided_backprop/results/'

    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    count = 0
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            to_image.paste(img_list[count], ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            count+=1
    return to_image.save(IMAGE_SAVE_PATH + filename)


if __name__ == '__main__':
    nrows = 9854
    SOURCE = '/mnt/xuyuhang/pytorch-cnn-visualizations/dataset_10_1000/'
    pretrained_model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=True)

    img_name_list = []

    # ground truth
    ground_truth_path = SOURCE + 'label_memmapped_new.dat'
    ground_truth_CE = find_GT_for_CE(ground_truth_path)

    # dataset_dim_list = [46656, 46656, 93312, 23328, 34992, 34992, 46656, 10816]  # squeeze
    # weight_dim_list = [16, 16, 32, 32, 48, 48, 64, 64]  # squeeze

    # dataset_dim_list = [186624, 186624, 373248, 93312, 139968, 139968, 186624, 43264]  # expand
    # weight_dim_list = [64, 64, 128, 128, 192, 192, 256, 256]  # expand

    dataset_dim_list = [373248, 373248, 746496, 186624, 279936, 279936, 373248, 86528] #concat
    weight_dim_list = [128, 128, 192, 192, 256, 256, 512, 512] # concat

    # 0, 1, 2, 3, 4, 5, 6, 7
    # 3, 4, 5, 7, 8, 9, 10, 12
    layer_index_list = (3, 4, 5, 7, 8, 9, 10, 12)
    layer_name = 'concat'
    # layer_index = 7
    for layer_index in range(8):
        cnn_layer = layer_index_list[layer_index]
        print('------------ layer '+str(layer_index+1)+' '+layer_name+'-------------')
        activation_path = SOURCE + layer_name + '_activation_fire' + str(layer_index + 1) + '.dat'


        layer_topx_img_path, filter_topx_index = find_topx_activation(activation_path, dataset_dim_list[layer_index],
                                                                      weight_dim_list[layer_index])
        print('filter_topx_index: ', filter_topx_index)
        preprocess_img_original_list = preprocess_images_original(layer_topx_img_path)
        save_images(preprocess_img_original_list, layer_name+'_fire'+str(layer_index+1)+'_org.jpg')

        gbp_img_list = []
        preprocess_img_model_list = preprocess_image_model(layer_topx_img_path)
        preprocess_img_model_list = np.reshape(preprocess_img_model_list, (-1, 9))
        for index, filter_index in enumerate(filter_topx_index):
            gbp_img_list_filter = gbp_visualization(pretrained_model, preprocess_img_model_list[index],
                                                    cnn_layer, filter_index, layer_name)
            gbp_img_list.extend(gbp_img_list_filter)
        # 2d list to 1d
        # gbp_img_list = sum(gbp_img_list, [])
        # print(len(gbp_img_list[0]))
        save_images(gbp_img_list, layer_name+'_fire' + str(layer_index + 1) + '_vis.jpg')
    print('Layer Guided backprop completed')
