df_train = pd.read_json('Modified_MaSaC_train_efr.json')
df_val = pd.read_json('Modified_MaSaC_val_efr.json')


# 定义一个安全地将标签转换为浮点数的函数
def safe_float_conversion(item):
    try:
        return float(item)
    except ValueError:
        return 0.0


# 应用此函数转换 train 和 val 的标签
df_train['triggers'] = df_train['triggers'].apply(lambda x: [safe_float_conversion(item) for item in x])
df_val['triggers'] = df_val['triggers'].apply(lambda x: [safe_float_conversion(item) for item in x])


# 定义用于组合对话和标签的函数，并填充标签到最大话语数量
def combine_dialogues_and_labels(dialogues, label_list, max_length):
    input_text = " [SEP] ".join(dialogues)
    labels_padded = label_list + [-1.0] * (max_length - len(label_list))  # 填充 -1.0 作为浮点数
    return input_text, labels_padded


# 确定最大话语数量
max_dialogue_length = max(max(len(dialogues) for dialogues in df_train['utterances']),
                          max(len(dialogues) for dialogues in df_val['utterances']))

# 更新训练和验证数据
train_texts, train_labels = [], []
val_texts, val_labels = [], []
for text, label in zip(df_train['utterances'], df_train['triggers']):
    combined_text, combined_label = combine_dialogues_and_labels(text, label, max_dialogue_length)
    train_texts.append(combined_text)
    train_labels.append(combined_label)

for text, label in zip(df_val['utterances'], df_val['triggers']):
    combined_text, combined_label = combine_dialogues_and_labels(text, label, max_dialogue_length)
    val_texts.append(combined_text)
    val_labels.append(combined_label)