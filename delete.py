import random
# 定义文件路径
train_file = "WN18RR_original/train.txt"
lowest_scores_file = "output.txt"
output_file = "train.txt"

# 读取文件内容
def read_file(filepath):
    with open(filepath, "r", encoding='UTF-8') as file:
        return [line.strip() for line in file.readlines()]

# 将过滤后的数据写入新文件
def write_to_file(filepath, data):
    with open(filepath, "w", encoding='UTF-8') as file:
        file.write("\n".join(data))

'''最优三元组最优路径'''
# 主程序

def filter_train_data(train_file, lowest_scores_file, output_file):
    # 读取文件数据
    train_data = read_file(train_file)
    lowest_scores_data = read_file(lowest_scores_file)

    print(f"Train data: {len(train_data)}")
    print(f"Lowest scores data: {len(lowest_scores_data)}")

    # 将lowest_scores_data转为集合，方便查找
    lowest_scores_set = set(lowest_scores_data)

    # 过滤train_data中的元素，去除与lowest_scores_set中相同的元素
    filtered_data = [line for line in train_data if line not in lowest_scores_set]
    print(len(filtered_data))
    if filtered_data:
        write_to_file(output_file, filtered_data)
        print(f"过滤后的结果已保存到: {output_file}")
        print(f"过滤后的结果大小: {len(filtered_data)}")
    else:
        print("没有任何元素符合过滤条件。")

# 执行过滤操作
filter_train_data(train_file, lowest_scores_file, output_file)
