from utils.include import *

class DBSCAN():
    def __init__(self, args, data_all, logger):
        # pdb.set_trace()
        self.samples = data_all['samples']
        self.score_matrix = data_all['score_matrix']
        self.eps = args.eps
        self.min_pts = args.min_pts
        self.log = logger

    def fit(self):
        # 获得数据的行和列(一共有n条数据)
        n, m = self.samples.shape
        # 将矩阵的中小于self.min_pts的数赋予1，大于self.min_pts的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
        core_points_idx = np.where(np.sum(np.where(self.score_matrix <= self.eps, 1, 0), axis=1) >= self.min_pts)[0]
        # 初始化类别，0代表未分类。
        cluster_labels = np.full((n,), 0)
        cur_label = 1
        # 遍历所有的核心点
        for point_idx in core_points_idx:
            # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
            if (cluster_labels[point_idx] == 0):
                # 首先将点point_idx标记为当前类别(即标识为已操作)
                cluster_labels[point_idx] = cur_label
                # 然后寻找种子点的self.eps邻域且没有被分类的点，将其放入种子集合
                neighbour=np.where((self.score_matrix[:, point_idx] <= self.eps) & (cluster_labels==0))[0]
                seeds = set(neighbour)
                # 通过种子点，开始生长，寻找密度可达的数据点，一直到种子集合为空，一个簇集寻找完毕
                while len(seeds) > 0:
                    # 弹出一个新种子点
                    new_point = seeds.pop()
                    # 将new_point标记为当前类
                    cluster_labels[new_point] = cur_label
                    # 寻找new_point种子点self.eps邻域（包含自己）
                    query_result = np.where(self.score_matrix[:,new_point]<=self.eps)[0]
                    # 如果new_point属于核心点，那么new_point是可以扩展的，即密度是可以通过new_point继续密度可达的
                    if len(query_result) >= self.min_pts:
                        # 将邻域内且没有被分类的点压入种子集合
                        for resultPoint in query_result:
                            if cluster_labels[resultPoint] == 0:
                                seeds.add(resultPoint)
                # 簇集生长完毕，寻找到一个类别
                cur_label = cur_label + 1
        return cluster_labels
