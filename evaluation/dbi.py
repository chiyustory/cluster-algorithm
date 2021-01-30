#!coding:utf-8
from utils.include import *
from scipy.spatial.distance import cdist

class DBI():
    def __init__(self, args, data_all, logger):   
        self.true_labels = data_all['labels']
        self.samples = data_all['samples']
        self.sample_num = len(self.true_labels)
        self.log = logger

    def euclidean_dist(self, vec1, vec2):
        return np.sqrt(np.sum(np.square(vec1 - vec2)))

    # 计算两个矩阵的距离矩阵
    def compute_distances(self, A, B):
        return cdist(A,B,metric='euclidean')


    def avg_dist_intra_cluster(self, cluster):
        """Computes average intra-cluster distance which is neccessary to computer
        DBI.
        Args:
            cluster: An cluster = {x1, x2,..., xk}. xi is a row.
        Returns:
            A float number of average intra-cluster distance.
        """
        sigma = 0
        size = cluster.shape[0]
        for i in range(size):
            for j in range(size):
                if i < j:
                    sigma += self.euclidean_dist(cluster[i], cluster[j])
        return 2 * sigma / (size * (size - 1))


    def center_point(self, cluster):
        center = np.sum(cluster, axis=0) / cluster.shape[0]
        return center


    def dist_inter_cluster(self, cluster1, cluster2):
        """Computes the distance of center points of two clusters.
        Args:
            cluster1, cluster2: An cluster = {x1, x2,..., xk}. xi is a row.
        Returns:
            A float number of average inter-cluster distance.
        """
        center1 = self.center_point(cluster1)
        center2 = self.center_point(cluster2)
        distance = self.euclidean_dist(center1, center2)
        return distance


    def dbi(self, cluster1, cluster2):
        """Computes the Davies-Boudlin Index of two clusters.
        Args:
            cluster1, cluster2: An cluster = {x1, x2,..., xk}. xi is a row.
        Returns:
            A float number of Davies-Boudlin Index.
        """
        index = ((self.avg_dist_intra_cluster(cluster1) + 
                self.avg_dist_intra_cluster(cluster2)) / 
                self.dist_inter_cluster(cluster1, cluster2))
        return index
    

    def convert_labels(self, labels):
        label_idxs_dict = {}
        idx_label_dict = {}
        for idx,label in enumerate(labels):
            idx_label_dict[idx] = label
            if label not in label_idxs_dict:
                label_idxs_dict[label] = []
            else:
                label_idxs_dict[label].append(idx)
        return idx_label_dict, label_idxs_dict

    def fit(self, cluster_labels):
        idx_label_dict, label_idxs_dict = self.convert_labels(cluster_labels)
        cluster_num = len(label_idxs_dict)
        dbi = 0.0
        for i, temp1 in enumerate(label_idxs_dict.items()):
            max_index = 0.0
            index = 0.0
            for j, temp2 in enumerate(label_idxs_dict.items()):
                print("{}/{}".format(i,j))
                if i != j:
                    cluster1 = self.samples[temp1[1]]
                    cluster2 = self.samples[temp2[1]]
                    index = self.dbi(cluster1, cluster2)
                    if max_index < index:
                        max_index = index
            dbi += max_index
        dbi /= cluster_num
        self.log.info("dbi: " + "{0:.3f}".format(dbi))