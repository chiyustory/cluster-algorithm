#!coding:utf-8
from utils.include import *
# import numpy as np
# import pdb


def create_log(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args.save_dir, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 日志同时显示在终端和log中
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

# 将分类后的数据可视化显示
def plot_data(args, samples, labels):
    label_num = len(set(labels))
    print("Label num is {}".format(label_num))
    fig = plt.figure()
    colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(0,label_num):
        select_color = colors[i % len(colors)]
        # pdb.set_trace()
        cur_cluster = samples[np.where(labels==i)]
        ax.scatter(cur_cluster[:,0], cur_cluster[:,1], c=select_color, s=12)
    # plt.savefig(os.path.join(args.save_dir,'data.jpg'))
    plt.savefig(os.path.join(args.save_dir,'data_cluster.jpg'))


# 归一化 [0,1]
def normalized(samples):
    min_val = np.min(samples, axis=0)
    max_val = np.max(samples, axis=0)
    # pdb.set_trace()
    return (samples - min_val) / float(max_val - min_val)


# 标准化
def standard(samples):
    mu = np.mean(samples, axis=0)
    sigma = np.std(samples, axis=0)
    return (samples - mu) / sigma


if __name__ == '__main__':
    x = np.arange(12).reshape((3, 4))
    print(x)
    x1 = normalized(x)
    print(x1)
    print(x1.max(), x1.min())
    x2 = standard(x)
    print(x2)
    print(x2.max(), x2.min())
