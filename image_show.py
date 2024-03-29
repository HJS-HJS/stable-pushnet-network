import numpy as np
import matplotlib.pyplot as plt
import torch

_start_idx = 300
_jump = 50
_pic_num = 4

fig = plt.figure()
for i in range(_pic_num * _pic_num):
    idx = str(_start_idx + i * _jump)
    np_image = np.load('./data/tensors/image_' + idx.zfill(7) + '.npy')
    # print(type(np_image))
    # print(np_image.shape)
    
    np_image = torch.from_numpy(np_image)
            
    image = np_image.permute(0, 1)
    ax = fig.add_subplot(_pic_num, _pic_num, i + 1)
    ax.imshow(image)

# plt.imshow(image)
plt.show()