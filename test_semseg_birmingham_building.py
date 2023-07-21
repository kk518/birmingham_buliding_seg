## 导入测试集 模型推理 生成标签  保存文件
import argparse

import numpy as np
from utils.helper_ply import write_ply
from  data_utils.birminghamDataLoader import  BMHDataset_test
import  os
import importlib
import  torch
import sys
from pathlib import Path
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
  #  parser.add_argument('--log_dir', default="building_sem"  ,type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=2, help='area for testing, option: 1-6 [default: 5]')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   # experiment_dir = 'log/sem_seg/building_seg' + args.log_dir
    experiment_dir = 'log/sem_seg/building_seg/building_seg_pointnet'

    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    Test_area = args.test_area

    ALL_pred_label = []
    ALL_choice_point = []

# 加载数据
    data_root = './data/birmingham_NO_lable'
    num_point, test_area, block_size, sample_rate = 512, 2, 1.0, 1

    point_data = BMHDataset_test(split='test', data_root=data_root, num_point=num_point, test_area=test_area,block_size=block_size, sample_rate=sample_rate)


    # 加载模型
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load( './log/sem_seg/2023-07-15_20-17/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    testDataLoader = torch.utils.data.DataLoader(point_data, batch_size= BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=False)

    print("开始推理")
    with torch.no_grad():
        num_batches = len(testDataLoader)
        for i, (points,select_points) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            select_points = select_points.data.numpy()
            select_points = select_points.reshape(-1,6)
            ALL_choice_point.append(select_points)

            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.float().cuda()
            points = points.transpose(2, 1)

            seg_pred, _= classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            pred_choice = seg_pred.contiguous().cpu().data.max(1)[1].numpy()

            ALL_pred_label.extend(pred_choice.tolist())

    print("推理完毕")

    ALL_pred_label = np.array(ALL_pred_label)
    ALL_pred_label = ALL_pred_label.reshape(-1,1)

    # 创建空的目标数组
    result = np.empty((0, 6))
    # 合并列表中的数组
    for arr in ALL_choice_point:
        result = np.concatenate((result, arr), axis=0)
    ALL_choice_points = result
    ALL_choice_points.reshape(-1,6)


    points_num = ALL_choice_points.shape[0]
    label_num  = ALL_pred_label.size

    num = int(int(points_num) - label_num)
    fill_list = [1] * num
    fill_list = np.array(fill_list).reshape(-1,1)
    ALL_pred_label= np.vstack((ALL_pred_label,fill_list))


    points_and_label = np.hstack( (ALL_choice_points,ALL_pred_label))

    for i in range(points_and_label.shape[0]):
        if points_and_label[i, -1] == 0:
            points_and_label[i, 3] = 255
            points_and_label[i, 4:6] = 0
        elif points_and_label[i, -1] == 1:
            points_and_label[i, 3:6] = 255

    points,rgb,labels = points_and_label[:, 0:3],points_and_label[:, 3:6],points_and_label[:, -1]

    out_path = str(visual_dir)+"\\birmingham_block_"+str(Test_area)+"_pointnet_256"
    print("写入ply")
    write_ply(out_path, [points, rgb, labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    print("ply写入完毕")




if __name__ == '__main__':
    args = parse_args()
    main(args)
