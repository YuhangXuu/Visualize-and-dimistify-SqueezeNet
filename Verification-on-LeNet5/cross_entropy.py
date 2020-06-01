import torch
from lenet import LeNet5
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd

import cv2
import os
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import keras
from sklearn.metrics import log_loss as LL
from torchvision.datasets.mnist import MNIST


class GradCAM:
    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    @classmethod
    def from_config(cls, arch: torch.nn.Module):
        target_layer = arch.maxpool2
        # target_layer = arch.features[12].expand3x3_activation
        return cls(arch, target_layer)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


def extract_image_onelayer(input_batch):
    l = model.maxpool2 # need to modify
    layer = []

    def hook(module, input, output):
        layer.append(output.clone().detach())

    handle = l.register_forward_hook(hook)
    with torch.no_grad():
        model(input_batch)
    handle.remove()
    return layer[0].cpu().numpy()[0].flatten()


def visualize_cam(mask, img, alpha=1.0):
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap


def create_memmap(file, col):
    filename = cam_dataset_path + file
    return np.memmap(filename, dtype=np.float32, mode='w+', shape=(nsamples, col))


def sliding_window_on_layer4(dataset_path, dataset_dim, window_len):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nsamples, dataset_dim))
    if np.isnan(np.sum(dataset)) == 1:
        print('spatial: nan')
        dataset = np.nan_to_num(dataset)

    dataset[dataset < 0.7] = 0
    # print('set 0 when value < 0.7')
    layer_output_len = int(np.sqrt(dataset_dim))
    dataset_matrix = dataset.reshape(nsamples, layer_output_len, -1)
    slides = layer_output_len - window_len + 1
    avg_pixel_max_window_activation_list = []
    spatial_pos = []
    for n in range(nsamples):
        if n % 1000 == 0:
            print('.', end="")
        activation_sum = np.sum(dataset_matrix[n, :, :])
        max_window_activation, row, col = 0, -1, -1
        for i in range(slides):
            for j in range(slides):
                window_activation = np.sum(dataset_matrix[n, i:i+window_len, j:j+window_len])
                if window_activation > max_window_activation:
                    max_window_activation = window_activation
                    row, col = i, j
        center_row = row + window_len // 2
        center_col = col + window_len // 2
        center_location = center_row * layer_output_len + center_col

        spatial_pos.append(center_location)
    # compare different window size
        avg_pixel_max_window_activation = max_window_activation / activation_sum
        avg_pixel_max_window_activation_list.append(avg_pixel_max_window_activation)
    avg_pixel_max_window_activation_array = np.array(avg_pixel_max_window_activation_list)
    spatial_pos_df = pd.DataFrame(spatial_pos)
    spatial_pos_df.to_csv(cam_dataset_path + 'spatial_pos_layer3_'
                          + str(window_len) + 'x' + str(window_len) + '_07.csv', index=False)
    return np.average(avg_pixel_max_window_activation_array)


def window_from_array(center_row, center_col, window_len, layer_output_len):
    dist = window_len // 2
    row_range = np.asarray([np.arange(a - dist, a + dist + 1) for a in center_row])
    col_range = np.asarray([np.arange(a - dist, a + dist + 1) for a in center_col])
    spatial_pos_list = []
    for i in range(nsamples):
        spatial_pos = []
        for r in row_range[i,:]:
            for c in col_range[i,:]:
                spatial_pos.append(r * layer_output_len + c)
        spatial_pos_list.append(spatial_pos)
    spatial_pos_array = np.asarray(spatial_pos_list)
    # print(spatial_pos_array.shape)
    return spatial_pos_array


def find_num_filters(arr, prob_cumsum):
    for index, num in enumerate(arr):
        if num > prob_cumsum:
            return index+1


def modify_spatial_pos(arr, layer_output_len, window_len):
    dist = window_len // 2
    arr[arr < dist] = dist
    arr[arr > layer_output_len - dist - 1] = layer_output_len - dist - 1
    return arr


def find_spatial_pos_RF(spatial_pos, layer_output_len, window_len):
    spatial_matrix_row = spatial_pos // layer_output_len
    spatial_matrix_col = spatial_pos % layer_output_len
    spatial_matrix_row = modify_spatial_pos(spatial_matrix_row, layer_output_len, window_len)
    spatial_matrix_col = modify_spatial_pos(spatial_matrix_col, layer_output_len, window_len)
    spatial_pos_array = window_from_array(spatial_matrix_row, spatial_matrix_col, window_len, layer_output_len)
    print('spatial dim:', spatial_pos_array.shape[1])
    return spatial_pos_array


def find_spatial_pos_alg(layer_index, layer, window_len, spatial_pos_next, layer_next, kennelNext, strideNext, paddingNext=0):
    rowNext = spatial_pos_next // layer_next
    colNext = spatial_pos_next % layer_next
    row = kennelNext // 2 + rowNext * strideNext - paddingNext
    col = kennelNext // 2 + colNext * strideNext - paddingNext
    row = modify_spatial_pos(row, layer, window_len)
    col = modify_spatial_pos(col, layer, window_len)
    spatial_pos = row * layer + col
    spatial_pos_df = pd.DataFrame(spatial_pos)
    spatial_pos_df.to_csv(cam_dataset_path + 'spatial_pos_layer' + str(layer_index)+'_3x3_07.csv', index=False)
    return rowNext, colNext, row, col, spatial_pos


def find_channel_pos(dataset_path, dataset_dim, layer_output_len, dataset_weight_dim, spatial_pos):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nsamples, dataset_dim))
    if np.isnan(np.sum(dataset)) == 1:
        print('spatial: nan')
        dataset = np.nan_to_num(dataset)

    # preprocessing dataset and dataset_weight
    dataset[dataset < 0] = 0
    size = layer_output_len * layer_output_len
    activation_list = []

    for c_pos in range(dataset_weight_dim):
        activation_sum = 0
        for i in range(nsamples):
            activation_sum += np.average(dataset[i, spatial_pos[i] + c_pos * size])
        activation_list.append(activation_sum)

    activation_array = np.array(activation_list)
    sorted_activation_sum = np.sort(activation_array)[::-1]
    sorted_index = np.argsort(activation_array)[::-1]
    sorted_activation_prob_cumsum = (sorted_activation_sum / np.sum(activation_array)).cumsum()
    num_filters = find_num_filters(sorted_activation_prob_cumsum, 0.95)
    print('spectral dim: ', num_filters)
    return sorted_index[:num_filters]


def find_subset(dataset_path, dataset_dim, layer_output_len, channel_pos, spatial_pos):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nsamples, dataset_dim))
    if np.isnan(np.sum(dataset)) == 1:
        print('spatial: nan')
        dataset = np.nan_to_num(dataset)
    size = layer_output_len * layer_output_len
    subset_index = []
    for c_pos in channel_pos:
        subset_index.append(spatial_pos + size * c_pos)
    subset_index = np.hstack(subset_index)
    subset = []
    for i in range(nsamples):
        subset.append(dataset[i, subset_index[i]])
    subset = np.vstack(subset)
    print('dataset_shrink dim:', subset.shape)
    return subset


def KMeans_Cross_Entropy(X, Y, num_class, num_bin=16):
    if np.unique(Y).shape[0] == 1:  # already pure
        return 0
    if X.shape[0] < num_bin:
        return -1
    kmeans = KMeans(n_clusters=num_bin, random_state=0).fit(X)
    prob = np.zeros((num_bin, num_class))
    for i in range(num_bin):
        idx = (kmeans.labels_ == i)
        tmp = Y[idx]
        for j in range(num_class):
            prob[i, j] = (float)(tmp[tmp == j].shape[0]) / len(kmeans.labels_[kmeans.labels_ == i])

    true_indicator = keras.utils.to_categorical(Y, num_classes=num_class)
    probab = prob[kmeans.labels_]

    return LL(true_indicator, probab)


def GT(data_loader):
    ground_truth_CE = []
    for i, (images, labels) in enumerate(data_loader):
        # print(labels.tolist())
        ground_truth_CE.append(labels.tolist())
    return np.hstack(ground_truth_CE)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    cam_dataset_path = '.../cam_dataset/'
    print('test')

    model = LeNet5()
    model.load_state_dict(torch.load('.../model/lenet.pt'))
    model.cuda()
    nsamples = 10000

    data_test = MNIST('.../data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                      ]))
    data_test_loader = DataLoader(data_test, batch_size=1, num_workers=1)

    model.eval()

    # generate saliency map on layer4
    dataset_layer4_saliency = create_memmap('layer3_saliency.dat', 25)
    for i, (images, labels) in enumerate(data_test_loader):
        if i % 1000 == 0:
            print('.', end="")
        images, labels = images.cuda(), labels.cuda()
        cam = GradCAM.from_config(model)
        mask, _ = cam(images, labels)
        dataset_layer4_saliency[i, ] = mask.cpu().numpy().flatten()

    # window location on layer4 saliency map
    for window_len in range(1, 6):
        print(sliding_window_on_layer4(cam_dataset_path + 'layer3_saliency.dat', 25, window_len))

    # extract layer output
    # 4704, 1176, 1600, 400
    # dataset_layer1 = create_memmap('layer0_activation.dat', 4704)
    # dataset_layer2 = create_memmap('layer1_activation.dat', 1176)
    # dataset_layer3 = create_memmap('layer2_activation.dat', 1600)
    dataset_layer4 = create_memmap('layer3_activation.dat', 400)
    count = 0
    for i, (images, labels) in enumerate(data_test_loader):
        if i % 1000 == 0:
            print('.', end="")
        images, labels = images.cuda(), labels.cuda()
        # dataset_layer1[count, ] = extract_image_onelayer(images)
        # dataset_layer2[count, ] = extract_image_onelayer(images)
        # dataset_layer3[count, ] = extract_image_onelayer(images)
        dataset_layer4[count, ] = extract_image_onelayer(images)
        count += 1
    print(count)

    # generate spatial pos
    spatial_pos_layer_df = pd.read_csv(cam_dataset_path + 'spatial_pos_layer3_3x3_07.csv')
    spatial_pos_layer = spatial_pos_layer_df.to_numpy().flatten()
    find_spatial_pos_alg(2, 10, 9, spatial_pos_layer, 5, 2, 2)

    spatial_pos_layer_df = pd.read_csv(cam_dataset_path + 'spatial_pos_layer2_3x3_07.csv')
    spatial_pos_layer = spatial_pos_layer_df.to_numpy().flatten()
    find_spatial_pos_alg(1, 14, 13, spatial_pos_layer, 10, 5, 1)

    spatial_pos_layer_df = pd.read_csv(cam_dataset_path + 'spatial_pos_layer1_3x3_07.csv')
    spatial_pos_layer = spatial_pos_layer_df.to_numpy().flatten()
    find_spatial_pos_alg(0, 28, 25, spatial_pos_layer, 14, 2, 2)

    spatial_pos_layer_df = pd.read_csv(cam_dataset_path + 'spatial_pos_layer0_3x3_07.csv')
    spatial_pos_layer = spatial_pos_layer_df.to_numpy().flatten()

    # compute cross entropy
    layer_output_len_list = [28, 14, 10, 5]
    # ratio_RF = [1, 1, 1, 1]
    window_len_list = [25, 13, 9, 4]
    dataset_dim_list = [4704, 1176, 1600, 400]
    weight_dim_list = [6, 6, 16, 16]
    layers = ['layer0_activation.dat', 'layer1_activation.dat',
              'layer2_activation.dat', 'layer3_activation.dat']

    ground_truth_CE = GT(data_test_loader)
    print('reduced dataset')
    for layer_index, layer in enumerate(layers)[::-1]:
        print('layer:', layer_index)
        dataset_path = cam_dataset_path + layer
        # full dataset
        dataset_full = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nsamples, dataset_dim_list[layer_index]))

        # dataset after dimension reduction
        spatial_pos_layer_df = pd.read_csv(cam_dataset_path + 'spatial_pos_layer' + str(layer_index)+'_3x3_07.csv')
        spatial_pos_layer = spatial_pos_layer_df.to_numpy().flatten()
        spatial_pos = find_spatial_pos_RF(spatial_pos_layer, layer_output_len_list[layer_index],
                                          window_len_list[layer_index])
        channel_pos = find_channel_pos(dataset_path, dataset_dim_list[layer_index], layer_output_len_list[layer_index],
                                       weight_dim_list[layer_index], spatial_pos)
        dataset_reduced = find_subset(dataset_path, dataset_dim_list[layer_index], layer_output_len_list[layer_index],
                                    channel_pos, spatial_pos)
        print('dataset_reduced shape:', dataset_reduced.shape)

        # Compute CE and MI
        CE_full = KMeans_Cross_Entropy(dataset_full, ground_truth_CE, 10, num_bin=16)
        CE_reduced = KMeans_Cross_Entropy(dataset_reduced, ground_truth_CE, 10, num_bin=16)
        print("cross entropy of dataset_full: ", CE_full)
        print("cross entropy of dataset_reduced: ", CE_reduced)
