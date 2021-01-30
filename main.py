#!coding:utf-8
import argparse
from utils.include import *
from utils.functions import *
from utils.dataset import *
from algorithm.dbscan import *
from algorithm.kmeans import *
from algorithm.dbscan_opt import *
from evaluation.bcubed import *
from evaluation.dbi import *
def main(args):
    logger = create_log(args)
    if args.save_data:
        read_data_and_save(args, logger)

    data_all = load_saved_data(args, logger)

    if args.algorithm == 'DBSCAN':
        logger.info("Use algorithm of DBSCAN")
        dbscan = DBSCAN(args, data_all, logger)
        cluster_labels = dbscan.fit()
    elif args.algorithm == 'KMEANS':
        logger.info("Use algorithm of KMEANS")
        kmeans = KMEANS(args, data_all, logger)
        cluster_labels = kmeans.fit()
    elif args.algorithm == 'AGNES':
        logger.info("Use algorithm of AGNES")
        pass
    elif args.algorithm == 'DBSCAN_OPT':
        logger.info("Use algorithm of DBSCAN Optimize")
        dbscan_opt = DBSCAN_OPTIMIZE(args, data_all, logger)
        cluster_labels = dbscan_opt.fit()

    plot_data(args, data_all['samples'], cluster_labels)

    logger.info("Use BCubed metric for evaluation")
    bcubed = BCUBED(args, data_all, logger)
    bcubed.fit(cluster_labels)

    # logger.info("Use DBI for evaluation")
    # dbi = DBI(args, data_all, logger)
    # dbi.fit(cluster_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        type=str,
                        default='./data/data_normalized.txt')
    parser.add_argument('--save_data', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--distance',
                        type=str,
                        default='euclidean',
                        help='euclidean or consine')
    parser.add_argument('--algorithm', type=str, default='DBSCAN_OPT', help='DBSCAN || KMEANS || AGNES || DBSCAN_OPT')
    # DBSCAN
    parser.add_argument('--eps', type=float, default=0.0625)
    parser.add_argument('--min_pts', type=float, default=25)
    #KMEANS
    parser.add_argument('--cluster_k', type=int, default=4)
    parser.add_argument('--iter_num', type=int, default=1000)
    parser.add_argument('--min_diffe', type=float, default=0.01)
    #AGNES
    # parser.add_argument('--dis_thresh', type=float, default=0.8)
    # DBSCAN Optimize
    # parser.add_argument('--eps', type=float, default=0.0625)
    parser.add_argument('--eps_soft', type=float, default=0.0825)
    # parser.add_argument('--min_pts', type=float, default=25)
    args = parser.parse_args()
    main(args)