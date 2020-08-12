import argparse
from texttable import Texttable

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument('--dataset', type=str, default='wiki',
                        help = "datasets where you run Model.", 
                        choices=['cora', 
                                 'citeseer', 
                                 'pubmed', 
                                 'wiki',
                                 'ModelNet40',
                                 'ntu2012',
                                 # SDCN
                                 'usps',
                                 'hhar',
                                 'reut',
                                 'acm',
                                 'dblp',
                                 # 'cite',
                                 ])
    # params for cora & citeseer & pubmed & wiki
    parser.add_argument('--construct_hg', type=int, default=1, 
                        help = "contruct inc_matrix from adj_matrix \
                    (hyperedge is Consists of every node and its 1-order neighbour).")

    # params for ModelNet40 & ntu2012
    parser.add_argument('--use_mvcnn_feature', type=int, default=1, 
                        help = "use_mvcnn_feature")
    parser.add_argument('--use_gvcnn_feature', type=int, default=1, 
                        help = "use_gvcnn_feature")
    parser.add_argument('--use_mvcnn_feature_for_structure', type=int, default=1, 
                        help = "use_mvcnn_feature_for_structure")
    parser.add_argument('--use_gvcnn_feature_for_structure', type=int, default=1, 
                        help = "use_gvcnn_feature_for_structure")

    parser.add_argument('--max_iter', type=int, default=60, 
                        help = "max_iter")                        

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                    help = "Random seed for train-test split. Default is 42.")
    
    return parser.parse_args()

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())
