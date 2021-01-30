#!coding:utf-8
from utils.include import *
from utils.functions import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def cal_score_matrix(args, samples, logger):
    logger.info("Calculation distance matrix by {}".format(args.distance))
    if args.distance == 'consine':
        # 相似度转化到0-100区间
        # score_matrix = (np.dot(samples, np.transpose(samples))) * 50
        # score_matrix = np.dot(samples, np.transpose(samples))
        score_matrix = squareform(pdist(samples,metric='cosine'))
    if args.distance == 'euclidean':
        score_matrix = squareform(pdist(samples,metric='euclidean'))
    logger.info("Distance matrix max value {} | min value {}".format(score_matrix.max(), score_matrix.min()))
    return score_matrix

def read_data_and_save(args,logger,show_data=False):
    data_all = OrderedDict()
    # 读取数据
    rf = open(args.data_file, 'r')
    data_lines = rf.readlines()
    samples = []
    labels = []
    for idx, line in enumerate(data_lines):
        inf = line.strip().split(',')
        # pdb.set_trace()
        tmp = np.array(list(map(float, inf[:-1])))
        # tmp = normalized(tmp)
        samples.append(tmp)
        labels.append(int(inf[-1]))
    rf.close()
    samples = np.array(samples)
    labels = np.array(labels)
    if show_data:
        plot_data(args,samples,labels)
    data_all['samples'] = samples
    data_all['labels'] = labels
    # pdb.set_trace()
    # 计算相似度矩阵
    score_matrix = cal_score_matrix(args, samples, logger)
    data_all['score_matrix'] = score_matrix
    # 保存所有数据
    logger.info("Save all data")
    np.save(os.path.join(args.save_dir,'data_all.npy'), data_all)



def load_saved_data(args, logger):
    logger.info("Load all data")
    data_all = np.load(os.path.join(args.save_dir, 'data_all.npy'), allow_pickle=True)
    return data_all.item()
    
