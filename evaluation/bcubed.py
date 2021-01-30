#!coding:utf-8
from utils.include import *

class BCUBED():
	def __init__(self, args, data_all, logger):
		self.true_labels = data_all['labels']
		self.sample_num = len(self.true_labels)
		self.log = logger

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
		# 更改聚类标签和真实标签的格式
		cluster_idx_label, cluster_label_idxs = self.convert_labels(cluster_labels)
		true_idx_label, true_label_idxs = self.convert_labels(self.true_labels)
		
		precision,recall = 0.0,0.0
		# 遍历每个样本，每个样本统计一次其对应簇的精度和召回，最后对所有样本求平均
		for idx in range(0, self.sample_num):
			# print("{}/{}".format(idx+1,self.sample_num))
			cluster_label = cluster_idx_label[idx]
			cluster_idxs = np.array(cluster_label_idxs[cluster_label])
			true_label = true_idx_label[idx]
			true_idxs = np.array(true_label_idxs[true_label])
			# pdb.set_trace()
			# 聚类标签为噪声点
			if cluster_label == 0:
				precision += 1.0
				# 真实标签为噪声点
				if true_label == 0:
					recall += 1.0
				else:
					recall += 1.0 / len(true_idxs)
			# 聚类标签不为噪声点
			else:
				# 真实标签为噪声点
				if true_label == 0:
					precision += 1.0 / len(cluster_idxs)
					recall += 1.0
				# 真实标签不为噪声点
				else:
					positive_num = len(set(cluster_idxs)&set(true_idxs))
					# 精度=该簇的真正例数量/该簇预测的所有样本数量
					precision += float(positive_num) / float(len(cluster_idxs))
					# 召回=该簇的真正例数量/该样本对应的所有正例数量
					recall += float(positive_num) / float( len(true_idxs) )

		precision /= float(self.sample_num)
		recall /= float(self.sample_num)

		self.log.info("precision: " + "{0:.3f}".format(precision))
		self.log.info("recall:    " + "{0:.3f}".format(recall))