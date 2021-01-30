from utils.include import *
from scipy.spatial.distance import cdist

class KMEANS():
    def __init__(self, args, samples_all, logger):
        self.samples = samples_all['samples']
        self.log = logger
        self.cluster_k = args.cluster_k
        self.iter_num = args.iter_num
        self.min_diffe = args.min_diffe

    # 计算两个矩阵的距离矩阵
    def compute_distances(self, A, B):
        return cdist(A,B,metric='euclidean')

    def fit(self):
        #一共有多少条数据
        n = np.shape(self.samples)[0]
        # 从n条数据中随机选择K条，作为初始中心向量
        # center_idx是初始中心向量的索引坐标
        center_idx = random.sample(range(0, n), self.cluster_k)
        # 获得初始中心向量,k个
        center_points = self.samples[center_idx]
        # 计算self.samples到center_points的距离矩阵
        dist = self.compute_distances(self.samples, center_points)
        # axis=1寻找每一行中最小值都索引
        # squeeze()是将label压缩成一个列表
        cluster_labels = np.argmin(dist, axis=1).squeeze()
        # 初始化old J
        old_var = -0.0001
        # 计算new J
        new_var = np.sum(np.sqrt(np.sum(np.power(self.samples - center_points[cluster_labels], 2), axis=1)))
        # 迭代次数
        count=0
        # 当ΔJ大于容差且循环次数小于迭代次数，一直迭代。负责结束聚类
        while count < self.iter_num and abs(new_var - old_var) < self.min_diffe:
            old_var = new_var
            for i in range(self.cluster_k):
                # 重新计算每一个类别都中心向量
                center_points[i] = np.mean(self.samples[np.where(cluster_labels == i)], 0)
            # 重新计算距离矩阵
            dist = self.compute_distances(self.samples, center_points)
            # 重新分类
            cluster_labels = np.argmin(dist, axis=1).squeeze()
            # 重新计算new J
            new_var = np.sum(np.sqrt(np.sum(np.power(self.samples - center_points[cluster_labels], 2), axis=1)))
            # 迭代次数加1
            count+=1
        return cluster_labels                                                         
