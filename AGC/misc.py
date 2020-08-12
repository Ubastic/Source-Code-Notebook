import logging
import os
import time
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

def create_directory(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def setup_logger(name, args, logpth):
    logfile = '{}-{}-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'), name, args.dataset)
    logfile = os.path.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

    logging.info(args)

