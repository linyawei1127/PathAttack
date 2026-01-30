relation_files = {
    '_member_meronym': '_member_meronym.txt',
    '_hypernym': '_hypernym.txt',
    '_similar_to': '_similar_to.txt',
    '_synset_domain_topic_of': '_synset_domain_topic_of.txt'
}

# 打开原始txt文件
with open('test.txt', 'r') as file:
    lines = file.readlines()

# 遍历每一行，根据关系存储到对应的txt文件中
for line in lines:
    parts = line.split('\t')
    if len(parts) == 3:
        relation = parts[1]
        if relation in relation_files:
            with open(relation_files[relation], 'a') as relation_file:
                relation_file.write(line)

# 关闭文件
for file_name in relation_files.values():
    with open(file_name, 'a'):
        pass  # 创建文件（如果不存在）

print("行已根据关系存储到不同的txt文件中。")
