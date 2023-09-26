import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


# # new_points [B, 3+D, nsample,npoint]     选取的中心点new_xyz [B, npoint, C]
# def  LocSE(new_points,new_xyz):
#     new_points = new_points[:,:3,:,:]   #所有点仅包含坐标信息[B, 3(xyz) , nsample , npoint]
#     new_points = new_points.permute(2,0,1,3)  # [ nsample ,B, 3(xyz)  , npoint]
#     new_xyz= new_xyz.permute(0, 2, 1)    #   [B, C, npoint]
#     #xyz 扩展 中心点坐标
#
#     #中心点xyz减去周围点xyz坐标
#     for i in range():
#         # 将 C 维度减去 d 维度，并覆盖 new_points 的对应维度
#         new_points[:, :, i, :] -= new_xyz[:, :, i].unsqueeze(2)
#
#     return new_points
# new_points [B, 3+D, nsample,npoint]     选取的中心点new_xyz [B, npoint, C]
def relative_pos_encoding(new_xyz, new_points):

    neighbor_xyz = new_points[:,:3,:,:]
    neighbor_xyz = neighbor_xyz.permute(0, 3, 2, 1)      #[b,npoint, nsample,d]

    xyz_tile = new_xyz.unsqueeze(2).repeat(1, 1, neighbor_xyz.shape[2], 1)   # [b,npoint, nsample,d]
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = torch.sqrt(torch.sum(relative_xyz ** 2, dim=-1, keepdim=True))
    relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
    return relative_feature

if __name__ == '__main__':
    new_xyz = torch.ones([1, 128, 3])
    new_points = torch.ones([1, 6, 32, 128])
    relative_feature =  relative_pos_encoding(new_xyz,new_points)
    print(relative_feature.shape)

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    计算两组点之间的欧式距离
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


"将idx映射为真实点"
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index ,  [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

"取半径内的点" \
"少于要求点则复制最近的" \
"多余则取最近的一部分"
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]      原始点云
        new_xyz: query points, [B, S, 3]   选取关键点云
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape  # N原始点云数量
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) #[ B,S,N ]
    sqrdists = square_distance(new_xyz, xyz)  #原始点云与选取点云之间每个点的欧式距离 [B, N, M]
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

# def sample_and_group_relative_pos_encoding(npoint, radius, nsample, xyz, points, returnfps=False):
#     """
#     最远点采样以及相对位置编码
#     Input:
#         npoint: 采样数
#         radius: 半径
#         nsample: 半径内采样点数量
#         xyz: input points position data, [B, N, 3]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, npoint, nsample, 3]
#         new_points: sampled points data, [B, npoint, nsample, 3+D]
#     """
#     B, N, C = xyz.shape     #c为xyz信息   n为4096
#     S = npoint                  #采样数
#     fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]  //index [b,npoint]
#     new_xyz = index_points(xyz, fps_idx)      #降采样后选取的关键点
#     #通过关键点查询每个关键点周围的点
#     idx = query_ball_point(radius, nsample, xyz, new_xyz)  #[B, S, nsample]
#     grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
#     grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
#
#     if points is not None:
#         grouped_points = index_points(points, idx)
#         new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
#     else:
#         new_points = grouped_xyz_norm
#     if returnfps:
#         return new_xyz, new_points, grouped_xyz, fps_idx
#     else:
#         return new_xyz, new_points

def sample_and_group_all(xyz, points):       #降采样与分组
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]     #选取关键点
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):         #降采样
    # npoint 采样数, radius 半径,  nsample 半径内采样点数量 , in_channel 9 + 3输入通道 , mlp [32, 32, 64]  , group_all
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]     输入点的坐标信息  [ B,3,N]
            points: input points data, [B, D, N]               输入点的所有信息  [ B,6,N]
        Return:
            new_xyz: sampled points position data, [B, C, S]            #采样后输出点的坐标信息
            new_points_concat: sample points feature data, [B, D', S]  #  特征D被全连接放大为D'
        """
        xyz = xyz.permute(0, 2, 1)        #调整为[ B,N,C ]  C=3
        if points is not None:
            points = points.permute(0, 2, 1)   #调整为[ B, N, D ]  D= 3+3

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)             # 未使用降采样与分组   最远点采样 new_xyz ,  new_points
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]                             #采样点的位置坐标信息
        # new_points: sampled points data, [B, npoint, nsample, 3+D]                #所有分组的采样点信息   点数npoint* nsample  3+D 通道数
        new_points = new_points.permute(0, 3, 2, 1) # [B, 3+D, nsample,npoint]   #所有的点  采样关键点以及分组后的点



        #对局部group中的每个点逐点mlp   #
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        #先对每一个点做mlp再最大池化得到全局特征  最大池化是在nsample 维度上进行的
        new_points = torch.max(new_points, 2)[0]  # [B, D' ,npoint]
        new_xyz = new_xyz.permute(0, 2, 1)   #[B, C, npoint]
        return new_xyz, new_points


class PointNetSetAbstraction_new(nn.Module):  # 降采样
    # npoint 采样数, radius 半径,  nsample 半径内采样点数量 , in_channel 9 + 3输入通道 , mlp [32, 32, 64]  , group_all
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]      输入点的坐标信息  [ B,3,N]  1024
            points: input points data, [B, D, N]               输入点的所有信息  [ B,6,N]
        Return:
            new_xyz: sampled points position data, [B, C, S]            #采样后输出点的坐标信息
            new_points_concat: sample points feature data, [B, D', S]  #  特征D被全连接放大为D'
        """
        xyz = xyz.permute(0, 2, 1)  # 调整为[ B,N,C ]  C=3
        if points is not None:
            points = points.permute(0, 2, 1)  # 调整为[ B, N, D ]  D= 3+3

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)  # 未使用降采样与分组   最远点采样 new_xyz ,  new_points
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]                             #采样点的位置坐标信息
        # new_points: sampled points data, [B, npoint, nsample, 3+D]                #所有分组的采样点信息   点数npoint* nsample  3+D 通道数
        new_points = new_points.permute(0, 3, 2, 1)  # [B, 3+D, nsample,npoint]   #所有的点  采样关键点以及分组后的点

        #局部空间位置编码


        # 对局部group中的每个点逐点mlp

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 先对每一个点做mlp再最大池化得到全局特征  最大池化是在nsample 维度上进行的
        new_points = torch.max(new_points, 2)[0]  # [B, D' ,npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, npoint]
        return new_xyz, new_points

"msg 多半径 " \
"npoint 采样个数" \
"radius_list 采样半径" \
"nsample_list 半径内选择采样数" \
    #不均匀点云
""
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))      #采样后的点newxyz
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  #取半径内的点
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)  #去均值
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)            #拼接法向量与位置信息
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)     # 三个不同半径特征拼接
        return new_xyz, new_points_concat

"  全局点特征后 特征拼接 上采样"
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

