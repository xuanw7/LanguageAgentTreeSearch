import os
import json
import argparse

from hotpotqa import HotPotQATask
from models import gpt_usage
from lats import lats_search
from tot import dfs_search
from rap import mcts_search
from cr import cr_search
import logging


import pickle 
import numpy as np
import re

def run(args):
    # print("start task")
    task = HotPotQATask()
    # print('init end')
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    count = 0
    task_accs = []
    info = []
    terminate_count = []

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.algorithm == 'lats':
            state, value, all_nodes, reward, em, tn_count = lats_search(args, task, i, args.iterations, True)
            terminate_count.append(tn_count)
            print("problem index:",i)
            print("tn_count:",tn_count)

        elif args.algorithm == 'tot':
            state, value, all_nodes, reward, em, tn_count = dfs_search(args, task, i, args.iterations, True)
            terminate_count.append(tn_count)
            print("problem index:",i)
            print("tn_count:",tn_count)

        elif args.algorithm == 'rap':
            state, value, all_nodes, reward, em, tn_count = mcts_search(args, task, i, args.iterations, True)
            terminate_count.append(tn_count)
            print("problem index:",i)
            print("tn_count:",tn_count)
        elif args.algorithm == 'cr':
            state, value, all_nodes, reward, em, tn_count = cr_search(args, task, i, args.iterations, True)
            terminate_count.append(tn_count)
            print("problem index:",i)
            print("tn_count:",tn_count)

        else:
            raise Exception("Search algorithm option not valid")
         # log main metric
        if em is None:
            em = 0
        print(i, "correctness: ", em)
        task_accs.append(em)
        cnt_avg = sum(task_accs) / len(task_accs)
        print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
        #all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]

        with open('accuracy.txt','wb') as f:
            pickle.dump(task_accs, f)

        with open('terminate_count.txt','wb') as f:
            pickle.dump(terminate_count, f)

        
       
    n = args.task_end_index - args.task_start_index
    # print('usage_so_far', gpt_usage(args.backend))

    with open('accuracy.txt','wb') as f:
        pickle.dump(task_accs, f)

    with open('terminate_count.txt','wb') as f:
        pickle.dump(terminate_count, f)

    with open('accuracy.txt','rb') as f:
        test_accs = pickle.load(f)
        print(test_accs)

    with open('terminate_count.txt','rb') as f:
        test_terminate= pickle.load(f)
        print(test_terminate)

 



def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613'], default='gpt-3.5-turbo-0613')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--n_generate_sample', type=int, default=1)  
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=50)
    args.add_argument('--log', type=str)
    args.add_argument('--algorithm', type=str, choices=['lats', 'rap', 'tot'])

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)