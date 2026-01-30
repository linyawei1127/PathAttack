import argparse
import numpy as np
from collections import defaultdict

def read_file(file_path):
    """读取文件并解析数据"""
    paths = defaultdict(list)
    with open(file_path, 'r', encoding="UTF-8") as file:
        lines = file.readlines()
        current_weight = None
        for line in lines:
            line = line.strip()
            if line.startswith("weight"):
                current_weight = float(line.split(":")[1].strip())
            elif line:  # 确保不处理空行
                parts = line.split("\t")
                if len(parts) == 3:
                    # 将三个部分按照1, 2, 3的形式存储
                    paths[current_weight].append((parts[0], parts[1], parts[2]))
    return paths

def sum_weights(paths):
    """计算每个三元组的权重之和"""
    triple_weights = defaultdict(float)
    for weight, triples in paths.items():
        for triple in triples:
            triple_weights[triple] += weight
    return triple_weights

def write_results(results, output_file, top_k):
    """将结果写入文件，仅输出权重最高的前top_k个三元组"""
    # 按照权重排序，并取权重最高的前top_k个
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(sorted_results)
    with open(output_file, 'w', encoding="UTF-8") as file:
        for triple, weight in sorted_results:
            file.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

def main():
    """主程序"""
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="Process knowledge graph completion data")
    parser.add_argument('--file_path', type=str, help="Path to the input data file")
    parser.add_argument('--top_k', type=int, default=None, help="Number of top triples to select (default: 10% of the dataset)")
    
    args = parser.parse_args()
    
    file_path = args.file_path
    output_file = 'output.txt'  # 结果输出文件
    
    try:
        # 如果 top_k 没有指定，根据文件大小计算
        if args.top_k is None:
            with open(file_path, 'r', encoding="UTF-8") as file:
                total_lines = sum(1 for line in file)  # Count the number of lines in the file
            top_k = int(total_lines * 0.10)  # Default to 10% of total lines
            print(f"Setting top_k to 10% of total lines: {top_k}")
        else:
            top_k = args.top_k
            print(f"Using custom top_k: {top_k}")

        # 读取和处理文件
        paths = read_file(file_path)
        triple_weights = sum_weights(paths)
        write_results(triple_weights, output_file, top_k)
        print(f"Results have been written to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
