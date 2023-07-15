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
    print("有{}个文件".format(len(filenames)))

    for filename in filenames:
         print("正在处理第",format(filename))
         path = IMPORT_PATH+"/"+filename
         xyz, rgb = DP.read_ply_data(path, True, False)


         point_list = np.hstack((xyz,rgb))


         outportFileName= OUTPORT_PATH+"/"+filename.replace(".ply","")+".txt"
         np.savetxt(outportFileName, point_list, fmt="%.5f,%.5f,%.5f,%d,%d,%d",delimiter=' ')


