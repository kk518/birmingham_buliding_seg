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
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
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
    experiment_dir = 'log/sem_seg/building_seg/'

    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    Test_area = args.test_area

    ALL_pred_label = []

# 加载数据
    data_root = './data/birmingham_NO_lable'
    num_point, test_area, block_size, sample_rate = 4096, 2, 1.0, 0.01

    point_data = BMHDataset_test(split='test', data_root=data_root, num_point=NUM_POINT, test_area=2,block_size=block_size, sample_rate=1)


    # 加载模型
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load( './log/sem_seg/pointnet_sem_seg/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    testDataLoader = torch.utils.data.DataLoader(point_data, batch_size= BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=False)

    with torch.no_grad():
        num_batches = len(testDataLoader)
        for i, (points) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.float().cuda()
            points = points.transpose(2, 1)

            seg_pred, _= classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            ALL_pred_label.extend(pred_choice.tolist())

    num = int(point_data.block_points[0].size/6) - len(ALL_pred_label)
    fill_list = [1] * num
    ALL_pred_label.extend(fill_list)
    ALL_pred_label = np.array(ALL_pred_label).reshape(-1,1)
    points = point_data.block_points[0]
    point_and_label = np.hstack( (points,ALL_pred_label))

    condition1 = point_and_label[:, -1] == 1
    condition2 = point_and_label[:, -1] == 0
    point_and_label[:, 3:6] = np.where(condition1.reshape(-1, 1), 255, point_and_label[:, 3:6])
    point_and_label[:, 3:6] = np.where(condition2.reshape(-1, 1), 255, point_and_label[:, 3:6])
    point_and_label[:, -1] = np.where(condition2, 0, point_and_label[:, -1])

    points,rgb,labels = point_and_label[:, 0:3],point_and_label[:, 3:6],point_and_label[:, -1]

    out_path = str(visual_dir)+"\\birmingham_block_"+str(Test_area)
    write_ply(out_path, [points, rgb, labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    print(1)




if __name__ == '__main__':
    args = parse_args()
    main(args)
