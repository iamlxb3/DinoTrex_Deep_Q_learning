import os
import pickle
from sklearn.decomposition import PCA


PCA_data_folder = 'PCA_data'
PCA_data_file = os.listdir(PCA_data_folder)

PCA_data_file_path_list = [os.path.join(PCA_data_folder, x) for x in PCA_data_file]


PCA_list = []
for PCA_file in PCA_data_file_path_list:
    with open(PCA_file, 'r') as f:
        feature_list = f.readlines()[0].strip().split(',')
        feature_list = feature_list
        feature_list = [int(x) for x in feature_list]
        PCA_list.append(feature_list)


n_components = 5000
pca = PCA(n_components=n_components)



pca.fit(PCA_list)
tran1 = pca.transform(PCA_list[0])
print ("tran1: ", tran1[0])
print ("len_tran1: ", len(tran1[0]))



# new_PCA_list = pca.fit_transform(PCA_list)
#
# print ("new_PCA_list: ", new_PCA_list[0])
# print ("new_PCA_list_len: ", len(new_PCA_list[0]))
#
# tran1 = [PCA_list[2]]


#
# pickle.dump(pca, open("fb_PCA", "wb"))
