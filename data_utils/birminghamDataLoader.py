import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
import time

class  BMHDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=[9,10], block_size=1.0,
                 sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        blocks = sorted(os.listdir(data_root))
        blocks = [block for block in blocks if 'birmingham_' in block]  # 包含所有区块的列表
        if split == 'train':
            blocks_split = [block for block in blocks if not 'birmingham_block_{}'.format(test_area) in block]  # 训练集
        else:
            blocks_split = [block for block in blocks if  'birmingham_block_{}'.format(test_area) in block]    # 测试集

        self.block_points, self.block_labels = [], []  # 一个区块所有点     与  对应的标签的列表
        self.block_coord_min, self.block_coord_max = [], []
        num_point_all = []  # 每个区域点的数量的列表
        labelweights = np.zeros(2)  # 所有房间内所有点的类别统计

        # 对每个房间分别处理
        for block_name in tqdm(blocks_split, total=len(blocks_split)):
            block_path = os.path.join(data_root, block_name)  # 房间路径

            block_data = np.loadtxt(block_path,delimiter=",")  # xyzrgbl, N*7

            points, labels = block_data[:, 0:6], block_data[:, 6]  # xyzrgb, N*6; l, N            #一个房间内的点与标签
            tmp, _ = np.histogram(labels,bins=2)  # 统计区间内每个元素出现的次数  tmp存储  也就是每个类别出现的次数   ？？
            labelweights += tmp  # 所有房间内所有点的类别统计
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]  # xyz轴上的最大   最小值
            self.block_points.append(points), self.block_labels.append(labels)
            self.block_coord_min.append(coord_min), self.block_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)  # 标签权重转换为32位浮点数
        labelweights = labelweights / np.sum(labelweights)  # 归一化
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)  # 标签值  每个元素相对于最大值的比例的三次方根
        print(self.labelweights)
        # 每个房间采样的采样次数
        sample_prob = num_point_all / np.sum(num_point_all)  # 每个块点数量占所有点的比例
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)  # 有多少个4096的个数
        block_idxs = []
        # [index] * int(round(sample_prob[index] * num_iter))这部分代码创建了一个长度为采样次数的列表，该列表中的元素都是对应房间的索引。
        # 通过循环遍历所有房间，将这些索引添加到block_idxs中。
        for index in range(len(blocks_split)):
            block_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))  # [  0,0,0,0,1,1,1,1 .....]   0代表在第0个房间采样 0的个数代表在0房间采样的次数
        self.block_idxs = np.array(block_idxs)
        print("Totally {} samples in {} set.".format(len(self.block_idxs), split))

#样本数为len(self.room_idxs)
    def __getitem__(self, idx):
        block_idx = self.block_idxs[idx]
        points = self.block_points[block_idx]   # N * 6
        labels = self.block_labels[block_idx]   # N
        N_points = points.shape[0]          #点的数量

        #在一个样本块中
        #给定点集中随机选择一个中心点，并根据中心点生成一个立方体范围。
        # 然后，从点集中筛选出位于立方体内部的点，要求满足筛选条件的点数量大于1024个才会跳出循环。
        while (True):
            center = points[np.random.choice(N_points)][:3]         #随机选择中心xyz坐标
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

#大于随机选择  小于设计复制
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize     归一化
        selected_points = points[selected_point_idxs, :]  # num_point * 6  选择出的点
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.block_coord_max[block_idx][0]         #x归一化后存储到6
        current_points[:, 7] = selected_points[:, 1] / self.block_coord_max[block_idx][1]         #y归一化
        current_points[:, 8] = selected_points[:, 2] / self.block_coord_max[block_idx][2]         #z归一化
        selected_points[:, 0] = selected_points[:, 0] - center[0]                                   #所有点在xy轴进行平移操作
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0  #rgb归一化
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        return current_points, current_labels

    def __len__(self):
        return len(self.block_idxs)


#无标签数据加载
class  BMHDataset_test(Dataset):
    def __init__(self, split='test ', data_root='trainval_fullarea', num_point=256, test_area=18, block_size=1.0,
                 sample_rate=1.0):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.sample_rate = sample_rate
        blocks = sorted(os.listdir(data_root))
        blocks = [block for block in blocks if 'birmingham_' in block]  # 包含所有区块的列表

        block_name = [block for block in blocks if  'birmingham_block_{}'.format(test_area) in block]    # 测试集

        self.block_points = [] # 一个区块所有点


        block_path = os.path.join(data_root, block_name[0])  # 区块路径
        print("加载数据")
        start_time = time.time()
        block_data = np.loadtxt(block_path,delimiter=",")  # xyzrgb  N*6
        end_time = time.time()
        print("加载数据完成用时："+str(end_time-start_time)+"s")

        #block_data = np.random.random((500000,6))
        points = block_data[:, 0:6]  # xyzrgb, N*6;         #一个区块内的点
        self.block_points.append(points)                          #一个区块内的点

        self.coord_min, self.coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]  # xyz轴上的最大   最小值


#一个区域采样次数
        self.num_iter = int(block_data.shape[0]* sample_rate / num_point)  # 有多少个num_point的个数

        print("Totally {} samples in {} set.".format(self.num_iter, split))

#样本数为len(self.room_idxs)
    def __getitem__(self, idx):


        points = self.block_points[0]  # N * 6

        N_points = points.shape[0]  # 点的数量
        #print("点的数量："+str(N_points))
        while (True):
            center = points[np.random.choice(N_points)][:3]         #随机选择中心xyz坐标
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size >= 512:
                break
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)



        # normalize     归一化
        select_points =  points[selected_point_idxs, :]
        selected_points = points[selected_point_idxs, :] # num_point * 6  选择出的点
        current_points = np.zeros((self.num_point, 9))  # num_point * 9

        current_points[:, 6] = selected_points[:, 0] / self.coord_max[0]  # x归一化后存储到6
        current_points[:, 7] = selected_points[:, 1] / self.coord_max[1] # y归一化
        current_points[:, 8] = selected_points[:, 2] / self.coord_max[2] # z归一化
        selected_points[:, 0] = selected_points[:, 0] - center[0]  # 所有点在xy轴进行平移操作
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0  # rgb归一化
        current_points[:, 0:6] = selected_points


        return  current_points , select_points

    def __len__(self):
        return  self.num_iter





if __name__ == '__main__':
    data_root = '../data/birmingham_NO_lable'
    num_point, test_area, block_size, sample_rate = 4096, 18 , 1.0, 1

    point_data = BMHDataset_test(split='test', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=1)

    import torch

    train_loader =   torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)


    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)

    num_point, test_area, block_size, sample_rate = 4096, 8, 1.0, 1
    data_root = '../data/birmingham_npy'
    point_data = BMHDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()