from helper import *
from data_loader import *
# sys.path.append('./')
from model.models import *
from torch.utils.data import DataLoader

class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format. 

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)
        
        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:         The dataloader for different data splits

        """
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train': 
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        # 注意：训练集这里存储的格式用于训练，不同于评估所需的 head/tail 格式
        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train':       get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head':  get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail':  get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head':   get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail':   get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructs the adjacency matrix for GCN.
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class.
        """
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters())

    def add_model(self, model, score_func):
        """
        Creates and initializes the computational graph for the given model.
        """
        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'compgcn_transe':
            model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_distmult':
            model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_conve':
            model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the model parameters.
        """
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Moves the batch tensors to the correct device.
        """
        triple, label = [x.to(self.device) for x in batch]
        return triple[:, 0], triple[:, 1], triple[:, 2], label

    def load_model(self, load_path):
        """
        Loads a saved model from the given path.
        """
        state = torch.load(load_path, map_location=torch.device('cpu'))
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def fit(self):
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.dataset)
        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        # 评估测试集等操作…
    
    def write_best_triples_train(self, output_file='best_triples_train.txt'):
        """
        挑选出所有在训练集中，head 和 tail 两个方向上排名均为 1 的三元组，
        并将对应的原始三元组 (s, r, o) 写入到指定的 txt 文件中。
        """
        self.model.eval()
        head_ranks = {}  # 用于存储 head 模式下各三元组的排名
        tail_ranks = {}  # 用于存储 tail 模式下各三元组的排名

        # 构造训练集的 head 和 tail 评估数据
        train_head_list = []
        train_tail_list = []
        for sub, rel, obj in self.data['train']:
            rel_inv = rel + self.p.num_rel
            train_tail_list.append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
            train_head_list.append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        # 创建 DataLoader（这里使用 TestDataset 格式，便于调用 collate_fn）
        train_head_loader = DataLoader(
            TestDataset(train_head_list, self.p),
            batch_size=self.p.batch_size,
            shuffle=False,
            num_workers=max(0, self.p.num_workers),
            collate_fn=TestDataset.collate_fn
        )
        train_tail_loader = DataLoader(
            TestDataset(train_tail_list, self.p),
            batch_size=self.p.batch_size,
            shuffle=False,
            num_workers=max(0, self.p.num_workers),
            collate_fn=TestDataset.collate_fn
        )

        # 处理 head 模式：输入格式为 (o, r_inv, s)
        with torch.no_grad():
            for batch in train_head_loader:
                triple_tensor, label = [x.to(self.device) for x in batch]
                pred = self.model.forward(triple_tensor[:, 0], triple_tensor[:, 1])
                b_range = torch.arange(pred.size(0), device=self.device)
                target_pred = pred[b_range, triple_tensor[:, 2]]
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 1e7, pred)
                pred[b_range, triple_tensor[:, 2]] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, triple_tensor[:, 2]]
                triple_list = triple_tensor.tolist()
                for triple, rank in zip(triple_list, ranks.tolist()):
                    # triple 格式为 (o, r_inv, s)
                    o, r_inv, s = triple
                    r = r_inv - self.p.num_rel  # 恢复原始关系编号
                    original_triple = (s, r, o)
                    head_ranks[original_triple] = rank

            # 处理 tail 模式：输入格式为 (s, r, o)
            for batch in train_tail_loader:
                triple_tensor, label = [x.to(self.device) for x in batch]
                pred = self.model.forward(triple_tensor[:, 0], triple_tensor[:, 1])
                b_range = torch.arange(pred.size(0), device=self.device)
                target_pred = pred[b_range, triple_tensor[:, 2]]
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 1e7, pred)
                pred[b_range, triple_tensor[:, 2]] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, triple_tensor[:, 2]]
                triple_list = triple_tensor.tolist()
                for triple, rank in zip(triple_list, ranks.tolist()):
                    original_triple = tuple(triple)  # triple 格式为 (s, r, o)
                    tail_ranks[original_triple] = rank

        # 筛选出同时满足 Rank(head)==1 且 Rank(tail)==1 的三元组
        best_triples = [triple for triple in head_ranks 
                        if head_ranks[triple] == 1 and tail_ranks.get(triple, None) == 1]

        # 将筛选出的三元组写入到文件中（这里可利用 id2ent/id2rel 转为对应名称）
        with open(output_file, 'w') as f:
            for s, r, o in best_triples:
                s_name = self.id2ent[s] if hasattr(self, 'id2ent') else str(s)
                r_name = self.id2rel[r] if hasattr(self, 'id2rel') else str(r)
                o_name = self.id2ent[o] if hasattr(self, 'id2ent') else str(o)
                f.write(f"{s_name}\t{r_name}\t{o_name}\n")
        self.logger.info(f"Saved {len(best_triples)} training triples with Rank(head)=1 and Rank(tail)=1 to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',       default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data',       dest='dataset', type=str, default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model',      dest='model', default='compgcn', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('-opn',        dest='opn', default='corr', help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch',      dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-gamma',      type=float, default=40.0, help='Margin')
    parser.add_argument('-gpu',        type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',      dest='max_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-l2',         type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',         type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers',type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed',       dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-restore',    dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-bias',       dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-num_bases',  dest='num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',   dest='init_dim', default=100, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',    dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',  dest='embed_dim', default=None, type=int, help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',  dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',   dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',   dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',  dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop',  dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',        dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h',        dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt',   dest='num_filt', default=200, type=int, help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',     dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-logdir',     dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config',     dest='config_dir', default='./config/', help='Config directory')
    args = parser.parse_args()

    if not args.restore:
        args.name = args.dataset + '_' + args.opn + '-' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Runner(args)
    model.fit()
    model.write_best_triples_train(output_file='wn18rr_best_triples_train.txt')
