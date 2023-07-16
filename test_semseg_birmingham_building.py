## 导入测试集 模型推理 生成标签  保存文件
from  data_utils.birminghamDataLoader import  BMHDataset_test




#加载数据
data_root = '../data/birmingham_NO_lable'
num_point, test_area, block_size, sample_rate = 4096, 2, 1.0, 0.01

point_data = BMHDataset_test(split='test', data_root=data_root, num_point=num_point, test_area=2,
                             block_size=block_size, sample_rate=1)

#加载模型


