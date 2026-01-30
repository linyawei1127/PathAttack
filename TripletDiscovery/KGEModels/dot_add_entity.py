import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import operator

import json
import logging
import argparse 
import math
from pprint import pprint
import errno

import time

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch.autograd as autograd

from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils


def get_additions(model, train_data, test_data, budget):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))

    ent_emb = model.emb_e.weight
    rel_emb = model.emb_rel.weight

    triples_to_add = []
    for test_idx, test_trip in enumerate(test_data):
        test_trip = torch.from_numpy(test_trip).to(device)[None, :]
        test_s, test_r, test_o = test_trip[:, 0], test_trip[:, 1], test_trip[:, 2]

        if_o_emb = model.emb_e(test_o).squeeze(dim=1)
        is_s_emb = model.emb_e(test_s).squeeze(dim=1)
        cos_sim_o = torch.matmul(if_o_emb, ent_emb.T).squeeze()
        cos_sim_s = torch.matmul(is_s_emb, ent_emb.T).squeeze()
        # print(cos_sim_o)

        # Filter for (s, r, o_dash), i.e. ignore o_dash that already exist
        filter_o = train_data[np.where((train_data[:, 0] == test_s.item()) 
                                     & (train_data[:, 1] == test_r.item())), 2].squeeze()
        cos_sim_o[filter_o] = 1e6

        filter_s = train_data[np.where((train_data[:,0] == test_o.item()) 
                                                   & (train_data[:,1] == test_r.item())), 2].squeeze()
        cos_sim_s[filter_s] = 1e6

        # Sort and rank - smallest cosine similarity means largest cosine distance
        min_values_o, argsort_o = torch.sort(cos_sim_o, -1, descending=False)
        min_values_s, argsort_s = torch.sort(cos_sim_s, -1, descending=False)

        # Counter for the number of perturbations added for this test_trip
        perturbations_added = 0

        # add test_s.item(), test_r.item(), o_dash
        for bud_o in range(len(argsort_o)):
            if perturbations_added >= budget:
                break  # Stop if the budget for this test_trip is reached

            o_dash = argsort_o[bud_o][None, None]
            o_dash = o_dash.item()
            add_trip = [test_s.item(), test_r.item(), o_dash]
            # add_trip = [o_dash, test_r.item(), test_s.item()]
            # print(add_trip)
            # Check if the triple already exists in train_data or triples_to_add
            m = (np.isin(train_data[:, 0], [add_trip[0]]) 
               & np.isin(train_data[:, 1], [add_trip[1]]) 
               & np.isin(train_data[:, 2], [add_trip[2]]))
            if not np.any(m) and add_trip not in triples_to_add:
                triples_to_add.append(add_trip)
                perturbations_added += 1

        if test_idx % 100 == 0 or test_idx == test_data.shape[0] - 1:
            print(test_idx)
            logger.info('Processed test triple {0}'.format(str(test_idx)))
            logger.info('Time taken: {0}'.format(str(time.time() - start_time)))

    logger.info('Time taken to generate edits: {0}'.format(str(time.time() - start_time)))

    return triples_to_add


parser = utils.get_argument_parser()
parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
parser.add_argument('--attack-batch-size', type=int, default=-1, help='Batch size for processing neighbours of target')
#Budget, target-split and rand-run values are to select the attack dataset to load influential triples
parser.add_argument('--sim-metric', type=str, default='cos', help='Value of similarity metric to use')


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args.target_split = '0_100_1' # which target split to use 
# #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
# args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
# args.rand_run = 1 #  a number assigned to the random run of the experiment
args.seed = args.seed + (args.rand_run - 1) # default seed is 17


# Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)
rng = np.random.default_rng(seed=args.seed)

args.epochs = -1 #no training here
args.model = "transe"
model_path = 'saved_models/processed_WN18RR_original_transe_200_0.0_0.3_0.3.model'
log_path = 'logs/attack_logs/{5}_add_5_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                           args.target_split, args.budget, args.rand_run,
                                                                 args.sim_metric
                                                                )


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename = log_path
                       )
logger = logging.getLogger(__name__)


# load the data from target data
data_path = 'data/{0}'.format(args.data)

n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)

##### load data####
data  = utils.load_data(data_path, add=True)
train_data, valid_data, test_data, target_data = data['train'], data['valid'], data['test'], data['target']

inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
inp_f.close()
to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['lhs'].items()}
to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['rhs'].items()}

model = utils.load_model(model_path, args, n_ent, n_rel, device)

triples_to_add = get_additions(model, train_data, target_data, args.budget)

#remove duplicate entries
df = pd.DataFrame(data=triples_to_add)
df = df.drop_duplicates()
# print(df.shape)
trips_to_add = df.values
# print(trips_to_delete.shape)
num_duplicates = len(triples_to_add) - trips_to_add.shape[0]
# print(num_duplicates)


per_tr = np.concatenate((trips_to_add, train_data))


#remove duplicate entries
df = pd.DataFrame(data=per_tr)
df = df.drop_duplicates()
# print(df.shape)
per_tr_1 = df.values
# print(trips_to_delete.shape)
num_duplicates_1 = per_tr.shape[0] - per_tr_1.shape[0]
# print(num_duplicates)


logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
logger.info('Number of duplicate adversarial additions: {0}'.format(num_duplicates))
logger.info('Number of adversarial additions already in train data: {0}'.format(num_duplicates_1))


logger.info ('Length of original training set: ' + str(train_data.shape[0]))
logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


new_train = per_tr_1
num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]


dict_path = './data/{0}'.format(args.data)
with open(dict_path + "/entities_dict.json", 'r', encoding='utf-8') as f:
    entity_map = json.load(f)
    entity_map = {v: k for k, v in entity_map.items()}
    # print(entity_map)

with open(dict_path + "/relations_dict.json", 'r', encoding='utf-8') as f:
    relation_map = json.load(f)
    relation_map = {v: k for k, v in relation_map.items()}
    # print(relation_map)

with open('additions.txt', 'w') as out:
    for item in trips_to_add:
        entity1_id, relation_id, entity2_id = item
        
        # Convert IDs to their corresponding values
        entity1 = entity_map.get(int(entity1_id), -1)  # -1 if not found
        relation = relation_map.get(int(relation_id), -1)
        entity2 = entity_map.get(int(entity2_id), -1)
        
        # Write the converted values to the output file
        out.write(f"{entity1}\t{relation}\t{entity2}\n")
