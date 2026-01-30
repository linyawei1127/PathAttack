# 读取文件内容
def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return {line.strip() for line in file.readlines()}


# 计算两个文件的交集
def compute_intersection(file1, file2):
    # 读取文件内容
    data1 = read_file(file1)
    data2 = read_file(file2)
    # print(len(data1))
    # print(len(data2))
    # 计算交集
    intersection = data1 & data2
    
    with open(output_file, "w", encoding="utf-8") as file:
        for item in intersection:
            file.write(f"{item}\n")


    return intersection, len(intersection)


# 主程序
file1 = "TripletDiscovery/KGEModels/target_distmult_WN18RR_0.txt"  # 替换为您的第一个文件路径
file2 = "TripletDiscovery/CompGCN/wn18rr_best_triples_train.txt"  # 替换为您的第二个文件路径
# file2 = "FB15k-237_original/target_distmult_2_compgcn.txt"  # 替换为您的第二个文件路径


output_file = "target_WN18RR_distmult_compgcn.txt"  # 输出文件路径

intersection, intersection_count = compute_intersection(file1, file2)

print(f"交集数: {intersection_count}")