from tool import  DataProcessing as DP
import  os
import  numpy as np

#讲PLY文件转为txt格式每行为  xyz rgb l     nx7

IMPORT_PATH = "../data/birmingham"
OUTPORT_PATH = "../data/birmingham_npy"

def get_all_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
    return filenames

if __name__ == '__main__':
    filenames = get_all_filenames(IMPORT_PATH)
    for filename in filenames:
         path = IMPORT_PATH+"/"+filename
         xyz, rgb, labels = DP.read_ply_data(path, True, True)
         labels  = np.where(labels == 2, 0, 1)
         labels= labels.reshape(-1,1)
         point_list = np.hstack((xyz,rgb))
         point_list = np.hstack((point_list,labels))

         outportFileName= OUTPORT_PATH+"/"+filename.replace(".ply","")+".txt"
         np.savetxt(outportFileName, point_list, fmt="%.5f,%.5f,%.5f,%d,%d,%d,%d",delimiter=' ')


