import json
import os
import sys
import pprint
from tqdm import tqdm
import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util


def load_vocab(dataset):
    name = dataset.config_dict()["class"]
    name = name.split(".")[-1].lower()
    path = os.path.dirname(os.path.dirname(__file__))

    # Load entities and relations from JSON files
    # name = "YAGO3-10"
    entities_file = os.path.join(path, "data", name, "entities_dict.json")
    relations_file = os.path.join(path, "data", name, "relations_dict.json")

    with open(entities_file, "r") as f:
        entities_dict = json.load(f)

    with open(relations_file, "r") as f:
        relations_dict = json.load(f)

    # Map IDs to original entities/relations
    original_entity_map = {v: k for k, v in entities_dict.items()}
    original_relation_map = {v: k for k, v in relations_dict.items()}

    # Create vocab lists based on dataset's vocab attributes
    entity_vocab = [entities_dict[t] for t in getattr(dataset, "entity_vocab")]
    relation_vocab = [relations_dict[t] for t in getattr(dataset, "relation_vocab")]

    return entity_vocab, relation_vocab, original_entity_map, original_relation_map
    


'''保存原始代码'''
def visualize(solver, batch_samples, entity_vocab, relation_vocab, original_entity_map, original_relation_map, output_file="visualization_results.txt"):
    num_relation = len(relation_vocab)
    # unbind(-1) 操作是将 sample 沿着最后一个维度拆开，返回三个张量，每个张量的形状为 (batch_size,)，
    # 分别对应三元组的头实体索引 h_index、尾实体索引 t_index 和关系索引 r_index。
    # h_index, t_index, r_index = sample.unbind(-1)
    # inverse = torch.stack([t_index, h_index, r_index + num_relation], dim=-1)
    inverse = torch.stack([torch.stack([t, h, r + num_relation], dim=-1) for h, t, r in batch_samples])
    # print(inverse)
    # batch = sample.unsqueeze(0)
    
    vis_batch = torch.stack([batch_samples[i // 2] if i % 2 == 0 else inverse[i // 2] for i in range(2 * len(batch_samples))])
    # print(vis_batch)

    batch = torch.stack(batch_samples).to(solver.device)
    vis_batch = vis_batch.to(solver.device)

    solver.model.eval()
    with torch.no_grad():
        pred, target = solver.model.predict_and_target(batch)
        # solver.evaluate("test")
    if isinstance(target, tuple):
        mask, target = target
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        rankings = rankings.squeeze(0)
    else:
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum(pos_pred <= pred, dim=-1) + 1
    # print(rankings)
    paths, weights, num_steps = solver.model.visualize(vis_batch)
    batch = batch.tolist()
    rankings = rankings.tolist()
    # print(rankings)
    paths = paths.tolist()
    # print(paths)
    weights = weights.tolist()
    num_steps = num_steps.tolist()

    with open(output_file, "a") as f:
        for i in range(len(vis_batch)):
            h, t, r = vis_batch[i]
            h_token = original_entity_map[entity_vocab[h]]
            t_token = original_entity_map[entity_vocab[t]]
            r_token = original_relation_map[relation_vocab[r % num_relation]]
            # print(h_token, r_token, t_token)
            # WN18RR，ranking 5 count 3
            # FB15k237 ranking 15 count 15
            # Family ranking 5 count 6
            # Nell-995 ranking count 
            if cfg.dataset["class"] in "WN18RR":
                if rankings[i // 2][i % 2] > 5:
                    continue
            elif cfg.dataset["class"] in "FB15k237":
                if rankings[i // 2][i % 2] > 15:
                    continue
            # print(i)
            count = 1
            h_origin, t_origin, r_origin = h, t, r
            for path, weight, num_step in zip(paths[i], weights[i], num_steps[i]):
                if cfg.dataset["class"] in "WN18RR":
                    if count > 3:
                        break
                elif cfg.dataset["class"] in "FB15k237":
                    if count > 15:
                        break
                count = count + 1
                # print(count)
                # NBFNet 注释掉
                if weight == float("-inf") or weight < 0.8:
                    break
                triplets = []
                # print(path)
                h_path, t_path, r_path = path[0]
                if (h_path == h_origin) and (t_path == t_origin) and (r_path == r_origin):
                    continue
                #print(h ,t, r)
                #print(h_origin, t_origin, r_origin)
                for h, t, r in path[:num_step]:
                    h_token = original_entity_map[entity_vocab[h]]
                    t_token = original_entity_map[entity_vocab[t]]
                    r_token = original_relation_map[relation_vocab[r % num_relation]]
                    if r >= num_relation:
                        #r_token += "^(-1)"
                        triplets.append("%s\t%s\t%s" % (t_token, r_token, h_token))
                    else:
                        triplets.append("%s\t%s\t%s" % (h_token, r_token, t_token))
                f.write("weight: %g\n%s\n" % (weight, "\n".join(triplets)))




if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())
    seed_value = args.seed + comm.get_rank()

    # 打印计算出来的种子值
    print(f"Setting random seed to: {seed_value}")

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] not in ["FB15k237", "OGBLWikiKG2", "WN18RR", "YAGO310"]:
        raise ValueError("Visualization is not implemented for %s" % cfg.dataset["class"])

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    entity_vocab, relation_vocab, original_entity_map, original_relation_map = load_vocab(dataset)
    
    #print(original_entity_map)
    # 测试整个数据集
    index = list(range(len(solver.test_set)))
    index = index[solver.rank::solver.world_size]
    output_file = "WN18RR_del_distmult_compgcn_astar.txt"

    batch_size = 32  # 批量处理大小，根据内存和性能调整
    # print(f"entity_vocab: {entity_vocab[0]}")
    for i in tqdm(range(0, len(index), batch_size), desc="Visualizing test set samples"):
        # if i >= 1:
        #     break
        batch_samples = [solver.test_set[j] for j in index[i:i+batch_size]]
        # print(batch_samples)
        visualize(solver, batch_samples, entity_vocab, relation_vocab, original_entity_map, original_relation_map, output_file=output_file)

