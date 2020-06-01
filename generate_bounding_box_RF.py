import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import keras
import math
from sklearn.metrics import log_loss as LL
import sys
import pandas as pd

sys.path.append('/mnt/xuyuhang/EDGE')
from EDGE_4_3_1 import EDGE
import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA


def find_num_filters(arr, prob_cumsum):
    for index, num in enumerate(arr):
        if num > prob_cumsum:
            return index+1


def modify_spatial_pos(arr, layer_output_len, window_len):
    dist = window_len // 2
    arr[arr < dist] = dist
    arr[arr > layer_output_len - dist - 1] = layer_output_len - dist - 1
    return arr


# def window_from_array_col(arr, window_len):
#     dist = window_len // 2
#     pos_range = np.asarray([np.arange(a - dist, a + dist + 1) for a in arr])
#     matrix_pos = np.tile(pos_range, window_len)
#     # print(matrix_pos.shape)
#     return matrix_pos
#
#
# def window_from_array_row(arr, window_len):
#     dist = window_len // 2
#     pos_range = [np.arange(a - dist, a + dist + 1) for a in arr]
#     matrix_pos = np.asarray([np.tile(pos.reshape(-1, 1), (1, window_len)).flatten() for pos in pos_range])
#     # print(matrix_pos.shape)
#     return matrix_pos


# def find_spatial_pos(dataset_path, dataset_dim, window_len):
#     dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, dataset_dim))
#     if np.isnan(np.sum(dataset)) == 1:
#         print('spatial: nan')
#         dataset = np.nan_to_num(dataset)
#     # print(dataset.shape)
#     spatial_pos_array = np.argmax(dataset, 1)
#     layer_output_len = int(np.sqrt(dataset_dim))
#     # print(spatial_pos_array.shape)
#     spatial_matrix_row = spatial_pos_array // layer_output_len
#     spatial_matrix_col = spatial_pos_array % layer_output_len
#
#     spatial_matrix_row_opt = modify_spatial_pos(spatial_matrix_row, layer_output_len, window_len)
#     spatial_matrix_col_opt = modify_spatial_pos(spatial_matrix_col, layer_output_len, window_len)
#
#     spatial_pos_array_opt = spatial_matrix_row_opt * layer_output_len + spatial_matrix_col_opt
#     # print('spatial dim:', spatial_pos_array_opt.shape)
#     return spatial_pos_array_opt


def sliding_window_on_fire9(dataset_path, dataset_dim, window_len):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, dataset_dim))
    if np.isnan(np.sum(dataset)) == 1:
        print('spatial: nan')
        dataset = np.nan_to_num(dataset)

    # dataset[dataset < 0.7] = 0
    print('set 0 when value < 0.7')
    layer_output_len = int(np.sqrt(dataset_dim))
    dataset_matrix = dataset.reshape(nrows, layer_output_len, -1)
    slides = layer_output_len - window_len + 1
    avg_pixel_max_window_activation_list = []
    spatial_pos = []
    for n in range(nrows):
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
    # spatial_pos_df = pd.DataFrame(spatial_pos)
    # spatial_pos_df.to_csv('/mnt/xuyuhang/pytorch-cnn-visualizations/dataset_10_1000/spatial_pos_Fire9_'
    #                       + str(window_len) + 'x' + str(window_len) + '_original.csv', index=False)
    return np.average(avg_pixel_max_window_activation_array)


def window_from_array(center_row, center_col, window_len, layer_output_len):
    dist = window_len // 2
    row_range = np.asarray([np.arange(a - dist, a + dist + 1) for a in center_row])
    col_range = np.asarray([np.arange(a - dist, a + dist + 1) for a in center_col])
    spatial_pos_list = []
    for i in range(nrows):
        spatial_pos = []
        for r in row_range[i,:]:
            for c in col_range[i,:]:
                spatial_pos.append(r * layer_output_len + c)
        spatial_pos_list.append(spatial_pos)
    spatial_pos_array = np.asarray(spatial_pos_list)
    # print(spatial_pos_array.shape)
    return spatial_pos_array


def find_spatial_pos_RF(spatial_pos_fire8, layer_output_len, window_len, ratio_RF):
    spatial_matrix_row = spatial_pos_fire8 // 13 * ratio_RF
    spatial_matrix_col = spatial_pos_fire8 % 13 * ratio_RF
    spatial_matrix_row = modify_spatial_pos(spatial_matrix_row, layer_output_len, window_len)
    spatial_matrix_col = modify_spatial_pos(spatial_matrix_col, layer_output_len, window_len)
    # spatial_array_row_opt = window_from_array_row(spatial_matrix_row_opt, window_len)
    # spatial_array_col_opt = window_from_array_col(spatial_matrix_col_opt, window_len)
    # spatial_pos_array_opt = spatial_array_row_opt * layer_output_len + spatial_array_col_opt
    spatial_pos_array = window_from_array(spatial_matrix_row, spatial_matrix_col, window_len, layer_output_len)
    print('spatial dim:', spatial_pos_array.shape[1])
    return spatial_pos_array


def find_channel_pos(dataset_path, dataset_dim, layer_output_len, dataset_weight_dim, spatial_pos):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, dataset_dim))
    if np.isnan(np.sum(dataset)) == 1:
        print('spatial: nan')
        dataset = np.nan_to_num(dataset)

    # preprocessing dataset and dataset_weight
    # dataset[dataset < 0] = 0
    size = layer_output_len * layer_output_len
    activation_list = []

    for c_pos in range(dataset_weight_dim):
        activation_sum = 0
        for i in range(nrows):
            activation_sum += np.average(dataset[i, spatial_pos[i] + c_pos * size])
        activation_list.append(activation_sum)

    # for c_pos in range(dataset_weight_dim):
    #     activation_sum = np.sum(dataset[:, c_pos * size: c_pos * size + size])
    #     activation_list.append(activation_sum)

    activation_array = np.array(activation_list)
    sorted_activation_sum = np.sort(activation_array)[::-1]
    sorted_index = np.argsort(activation_array)[::-1]
    sorted_activation_prob_cumsum = (sorted_activation_sum / np.sum(activation_array)).cumsum()
    num_filters = find_num_filters(sorted_activation_prob_cumsum, 0.95)
    print('spectral dim: ', num_filters)
    # plt.figure(figsize=(8, 5))
    # plt.plot(sorted_activation_prob_cumsum)
    # plt.savefig('/mnt/xuyuhang/pytorch-cnn-visualizations/cumsum_plot/concat_activation_fire'+str(layer_index+1)+'.jpg')
    return sorted_index[:num_filters]

# def channel_energy(dataset_path, dataset_dim, layer_output_len, dataset_weight_path, dataset_weight_dim,
#                    spatial_pos, spatial_reduction, weighted):
#     dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, dataset_dim))
#     dataset_weight = np.memmap(dataset_weight_path, dtype=np.float32, mode='c', shape=(nrows, dataset_weight_dim))
#     if np.isnan(np.sum(dataset)) == 1:
#         print('spatial: nan')
#         dataset = np.nan_to_num(dataset)
#     if np.isnan(np.sum(dataset_weight)) == 1:
#         print('spatial: nan')
#         dataset_weight = np.nan_to_num(dataset_weight)
#     # preprocessing dataset and dataset_weight
#     # dataset[dataset<0] = 0
#     dataset = np.abs(dataset)
#     dataset_weight = dataset_weight - dataset_weight.min()
#     dataset_weight /= dataset_weight.max()
#     size = layer_output_len * layer_output_len
#     activation_list = []
#     if spatial_reduction is True and weighted is True:
#         for c_pos in range(dataset_weight_dim):
#             activation_sum = 0
#             for i in range(nrows):
#                 activation_sum += np.average(dataset[i, spatial_pos[i] + c_pos * size]) * dataset_weight[i, c_pos]
#             activation_list.append(activation_sum)
#     elif spatial_reduction is True and weighted is False:
#         for c_pos in range(dataset_weight_dim):
#             activation_sum = 0
#             for i in range(nrows):
#                 activation_sum += np.average(dataset[i, spatial_pos[i] + c_pos * size])
#             activation_list.append(activation_sum)
#     elif spatial_reduction is False and weighted is True:
#         for c_pos in range(dataset_weight_dim):
#             activation_sum = 0
#             for i in range(nrows):
#                 activation_sum += np.average(dataset[i, c_pos * size]) * dataset_weight[i, c_pos]
#             activation_list.append(activation_sum)
#     elif spatial_reduction is False and weighted is False:
#         for c_pos in range(dataset_weight_dim):
#             activation_sum = 0
#             for i in range(nrows):
#                 activation_sum += np.average(dataset[i, c_pos * size])
#             activation_list.append(activation_sum)
#     activation_array = np.array(activation_list)
#
#     # print('activation_array: ', activation_array.shape)
#     sorted_activation_sum = np.sort(activation_array)[::-1]
#     sorted_activation_prob = sorted_activation_sum / np.sum(activation_array)
#     return sorted_activation_prob
#
#
# def channel_reduction_pca(dataset_path, dataset_dim, layer_output_len, dataset_weight_dim, spatial_pos):
#     dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, dataset_dim))
#     if np.isnan(np.sum(dataset)) == 1:
#         print('spatial: nan')
#         dataset = np.nan_to_num(dataset)
#     size = layer_output_len * layer_output_len
#     spatial_subset_index = []
#     for c_pos in range(dataset_weight_dim):
#         spatial_subset_index.append(spatial_pos + size * c_pos)
#     spatial_subset_index = np.hstack(spatial_subset_index)
#     spatial_subset = []
#     for i in range(nrows):
#         spatial_subset.append(dataset[i, spatial_subset_index[i]])
#     spatial_subset = np.vstack(spatial_subset)
#     print('spatial_subset dim:', spatial_subset.shape)
#
#     pca = PCA(n_components=2000)
#     # for i in range(nrows):
#     #     spatial_subset = dataset[i, spatial_subset_index[i]]
#     #     spatial_subset_2d = spatial_subset.reshape(window_len * window_len, dataset_weight_dim)
#     principalComponents = pca.fit_transform(spatial_subset)
#     energy = pca.explained_variance_ratio_
#     energy_cum = energy.cumsum()
#     prob_cumsum = 0.95
#     print('energy of ', prob_cumsum, ': ', find_num_filters(energy_cum, prob_cumsum))


def find_subset(dataset_path, dataset_dim, layer_output_len, channel_pos, spatial_pos):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, dataset_dim))
    if np.isnan(np.sum(dataset)) == 1:
        print('spatial: nan')
        dataset = np.nan_to_num(dataset)
    size = layer_output_len * layer_output_len
    subset_index = []
    for c_pos in channel_pos:
        subset_index.append(spatial_pos + size * c_pos)
    subset_index = np.hstack(subset_index)
    subset = []
    for i in range(nrows):
        subset.append(dataset[i, subset_index[i]])
    subset = np.vstack(subset)
    print('dataset_shrink dim:', subset.shape)
    return subset


def find_GT_for_CE(dataset_path):
    dataset = np.memmap(dataset_path, dtype=np.float32, mode='c', shape=(nrows, 1))
    dataset = np.nan_to_num(dataset)
    ground_truth_list = [284, 979, 511, 628, 320, 468, 460, 547, 194, 809]
    for index, gt in enumerate(ground_truth_list):
        dataset[dataset == gt] = index
    return dataset


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


# def compute_MI(matrix1, matrix2, flag):
#     # np.nan_to_num(matrix1, copy=False)
#     # np.nan_to_num(matrix2, copy=False)
#     # print('matrix1: ', matrix1.shape)
#     # print('matrix2: ', matrix2.shape)
#     # result = EDGE(matrix1, matrix2, U=20, gamma=[1, 0.001])
#     if flag == 0:
#         result = EDGE(matrix1, matrix2, U=20, gamma=[1.1, 1.1])
#     else:
#         result = EDGE(matrix1, matrix2, U=20, gamma=[1, 0.001])
#     return result


if __name__ == "__main__":
    print('num_bin = 32')
    print('spatial affects channel selection')

    nrows = 9854
    print('-----------------0.95, 7x7 07-----------------')
    layer_name = 'concat'
    print('-----------------' + layer_name + '-----------------')
    SOURCE = '/mnt/xuyuhang/pytorch-cnn-visualizations/dataset_10_1000/'
    ground_truth_path = SOURCE + 'label_memmapped_new.dat'
    ground_truth_list = [284, 979, 511, 628, 320, 468, 460, 547, 194, 809]
    layer_output_len_list = [54, 54, 54, 27, 27, 27, 27, 13]
    ratio_RF = [4, 4, 4, 2, 2, 2, 2, 1]
    # window_len_list = [37, 37, 37, 19, 19, 19, 19, 9] # 9x9
    window_len_list = [29, 29, 29, 15, 15, 15, 15, 9] # 7x7 feature map fixed
    # window_len_list = [21, 21, 21, 11, 11, 11, 11, 5] # 5x5 feature map fixed

    # dataset_dim_list = [46656, 46656, 93312, 23328, 34992, 34992, 46656, 10816]  # squeeze
    # weight_dim_list = [16, 16, 32, 32, 48, 48, 64, 64]  # squeeze

    # dataset_dim_list = [186624, 186624, 373248, 93312, 139968, 139968, 186624, 43264]  # expand
    # weight_dim_list = [64, 64, 128, 128, 192, 192, 256, 256]  # expand

    dataset_dim_list = [373248, 373248, 746496, 186624, 279936, 279936, 373248, 86528]  #concat
    weight_dim_list = [128, 128, 256, 256, 384, 384, 512, 512]  # concat

    # saliency_path_fire8 = \
    #     '/mnt/xuyuhang/pytorch-cnn-visualizations/dataset_10_1000/concat_activation_fire8_saliency.dat'
    # print(sliding_window_on_fire9(saliency_path_fire8, 169, 7))
    # print('finish')
    # for index in range(1,14):
    #     print('window_size: ', index)
    #     print(sliding_window_on_fire9(saliency_path_fire8, 169, index))
    #     print('\n')
    # print('set 0 when value < 0.6')

    spatial_pos_fire8_df = \
        pd.read_csv('/mnt/xuyuhang/pytorch-cnn-visualizations/dataset_10_1000/spatial_pos_Fire9_7x7_07.csv')
    spatial_pos_fire8 = spatial_pos_fire8_df.to_numpy().flatten()
    # print(spatial_pos_fire8.shape)

    # Input
    # input_path = '/mnt/xuyuhang/pytorch-cnn-visualizations/dataset_10_1000/image_input.dat'
    # input_spatial_pos = find_spatial_pos_RF(spatial_pos_fire8, 224, 121, 17)
    # input_channel_pos = np.array([0, 1, 2])
    # input_dataset_shrink = find_subset(input_path, 150528, 224, input_channel_pos, input_spatial_pos)
    # GT
    ground_truth_CE = find_GT_for_CE(ground_truth_path)

    # maxpooling ce and mi
    # dataset_dim_list_maxpooling = [279936, 186624, 86528]
    # layer_index_maxpooling = [1, 4, 8]
    # layer_output_len_list_maxpooling = [54, 27, 13]
    # window_len_list_maxpooling = [29, 15, 7]
    # ratio_RF_maxpooling = [4, 2, 1]
    # weight_dim_list_maxpooling = [96, 256, 512]
    # for index, layer_index in enumerate(layer_index_maxpooling):
    #     print('layer_maxpooling_', layer_index)
    #     dataset_path = SOURCE + 'maxpooling_' + str(layer_index) + '.dat'
    #     spatial_pos = find_spatial_pos_RF(spatial_pos_fire8, layer_output_len_list_maxpooling[index],
    #                                       window_len_list_maxpooling[index], ratio_RF_maxpooling[index])
    #     channel_pos = find_channel_pos(dataset_path, dataset_dim_list_maxpooling[index], layer_output_len_list_maxpooling[index],
    #                                    weight_dim_list_maxpooling[index], spatial_pos)
    #     dataset_shrink = find_subset(dataset_path, dataset_dim_list_maxpooling[index], layer_output_len_list_maxpooling[index],
    #                                  channel_pos, spatial_pos)
    #     #Compute CE and MI
    #     CE = KMeans_Cross_Entropy(dataset_shrink, ground_truth_CE, 10, num_bin=32)
    #     print("cross entropy: ", CE)
    #     # MI_output = EDGE(dataset_shrink, ground_truth_CE, L_ensemble=50)
    #     # MI_input = EDGE(input_dataset_shrink, dataset_shrink, U=15, L_ensemble=50)
    #     #
    #     # print("MI_input: ", MI_input)
    #     # print("MI_output: ", MI_output)

    # channel_energy_0_0_list = []
    # channel_energy_1_0_list = []
    # channel_energy_0_1_list = []
    # channel_energy_1_1_list = []
    for layer_index in range(8)[2:]:
        print('layer:', layer_index + 1)
        dataset_path = SOURCE + layer_name + '_activation_fire' + str(layer_index + 1) + '.dat'
        spatial_pos = find_spatial_pos_RF(spatial_pos_fire8, layer_output_len_list[layer_index],
                                          window_len_list[layer_index], ratio_RF[layer_index])
        channel_pos = find_channel_pos(dataset_path, dataset_dim_list[layer_index], layer_output_len_list[layer_index],
                                       weight_dim_list[layer_index], spatial_pos)
        dataset_shrink = find_subset(dataset_path, dataset_dim_list[layer_index], layer_output_len_list[layer_index],
                                     channel_pos, spatial_pos)

        # # Compute CE and MI
        CE = KMeans_Cross_Entropy(dataset_shrink, ground_truth_CE, 10, num_bin=32)
        # MI_output = compute_MI(dataset_shrink, ground_truth_CE, 1)
        # MI_input = compute_MI(input_dataset_shrink, dataset_shrink, 0)
        print("cross entropy: ", CE)
        # MI_output = EDGE(dataset_shrink, ground_truth_CE, L_ensemble=20)
        # MI_input = EDGE(input_dataset_shrink, dataset_shrink, U=15, L_ensemble=20)
        #
        # print("MI_input: ", MI_input)
        # print("MI_output: ", MI_output)

    #
    #     # save data for energy plotting
    #     channel_energy_0_0 = channel_energy(dataset_path, dataset_dim_list[layer_index],
    #                                         layer_output_len_list[layer_index], activation_weight_path,
    #                                         weight_dim_list[layer_index], spatial_pos, False, False)
    #     channel_energy_1_0 = channel_energy(dataset_path, dataset_dim_list[layer_index],
    #                                         layer_output_len_list[layer_index], activation_weight_path,
    #                                         weight_dim_list[layer_index], spatial_pos, True, False)
    #     channel_energy_0_1 = channel_energy(dataset_path, dataset_dim_list[layer_index],
    #                                         layer_output_len_list[layer_index], activation_weight_path,
    #                                         weight_dim_list[layer_index], spatial_pos, False, True)
    #     channel_energy_1_1 = channel_energy(dataset_path, dataset_dim_list[layer_index],
    #                                         layer_output_len_list[layer_index], activation_weight_path,
    #                                         weight_dim_list[layer_index], spatial_pos, True, True)
    #     channel_energy_0_0_list.append(channel_energy_0_0)
    #     channel_energy_1_0_list.append(channel_energy_1_0)
    #     channel_energy_0_1_list.append(channel_energy_0_1)
    #     channel_energy_1_1_list.append(channel_energy_1_1)
    # df_0_0 = pd.DataFrame(channel_energy_0_0_list)
    # df_1_0 = pd.DataFrame(channel_energy_1_0_list)
    # df_0_1 = pd.DataFrame(channel_energy_0_1_list)
    # df_1_1 = pd.DataFrame(channel_energy_1_1_list)
    # folder_name = '/mnt/xuyuhang/pytorch-cnn-visualizations/cumsum_plot/'
    # df_0_0.to_csv(folder_name + layer_name + '_0_0.csv', index=False)
    # df_1_0.to_csv(folder_name + layer_name + '_1_0.csv', index=False)
    # df_0_1.to_csv(folder_name + layer_name + '_0_1.csv', index=False)
    # df_1_1.to_csv(folder_name + layer_name + '_1_1.csv', index=False)


