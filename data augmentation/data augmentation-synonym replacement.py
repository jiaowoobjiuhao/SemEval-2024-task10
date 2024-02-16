import json
import nltk
from nltk.corpus import wordnet
import random

# 确保已经下载了 WordNet 数据
nltk.download('wordnet')

# 定义同义词替换函数
def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = [syn.lemmas()[0].name() for syn in wordnet.synsets(word)]
        if synonyms:
            synonym = random.choice(synonyms)
            new_sentence.append(synonym)
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence)

# 读取并处理 Modified_MaSaC_train_erc.json 文件
modified_file_path = 'Modified_MaSaC_train_erc.json'

# 读取 JSON 数组
with open(modified_file_path, 'r', encoding='utf-8') as file:
    modified_data = json.load(file)

# 对 utterances 进行同义词替换
for entry in modified_data:
    entry['utterances'] = [synonym_replacement(utterance) for utterance in entry['utterances']]

# 合并原始数据（ekphrasis_MaSaC_train_erc_back_translated.json）和处理过的数据
original_file_path = 'ekphrasis_MaSaC_train_erc_back_translated.json'
with open(original_file_path, 'r', encoding='utf-8') as file:
    original_data = [json.loads(line) for line in file]

combined_data = original_data + modified_data

# 将合并后的数据写入新文件
output_file_path = 'combined_ekphrasis_modified.json'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for item in combined_data:
        json.dump(item, file)
        file.write('\n')

print("Data combined and written to", output_file_path)
