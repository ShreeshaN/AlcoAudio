import numpy as np
from alcoaudio.utils.data_utils import read_npy

data_read_path = "/Users/pratcr7/Documents/GitHub/AlcoAudio/alco_data/"
train_data, train_labels = read_npy(data_read_path+"train_data.npy"), read_npy(data_read_path+"train_labels.npy")
# test_data, test_labels = read_npy(data_read_path+"test_data.npy"), read_npy(data_read_path+"test_labels.npy")

# index = []
# sz = np.shape(test_data)
# for i in range(sz[0]):
#     if (i!= 365 and i!= 480 and i!= 542 and i!= 632 and i!= 1023 and i!= 1107 and i!= 1926 and i!= 1936 and i!= 2272):
#         sh = np.shape(test_data[i])
#         if (sh[2] != 345):
#             index.append(i)




