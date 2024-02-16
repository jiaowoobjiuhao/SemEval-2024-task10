#subtask1
df_train = pd.read_json('combined_ekphrasis_modified.json',lines=True)
    #df_train = pd.read_json('Modified_MaSaC_train_erc.json')
    df_val = pd.read_json('Modified_MaSaC_val_erc.json')

    # 提取唯一的情绪标签并映射为数字
    unique_labels = sorted(set(label for label_list in df_train['emotions'] for label in label_list) |
                          set(label for label_list in df_val['emotions'] for label in label_list))
    emotion_to_int = {emotion: idx for idx, emotion in enumerate(unique_labels)}

    # 定义将情绪标签列表转换为整数列表的函数
    def labels_to_int(label_list, emotion_to_int_mapping):
        return [emotion_to_int_mapping[label] for label in label_list]

    # 定义用于组合对话和情绪标签的函数，并填充情绪标签到最大话语数量
    def combine_dialogues_and_labels(dialogues, label_list, max_length):
        input_text = " [SEP] ".join(dialogues)
        labels_int = labels_to_int(label_list, emotion_to_int)
        # 使用-1或其他指定值填充情绪标签列表，直到达到最大长度
        labels_int += [-1] * (max_length - len(labels_int))
        return input_text, labels_int

    # 确定最大话语数量
    max_dialogue_length = max(max(len(dialogues) for dialogues in df_train['utterances']),
                              max(len(dialogues) for dialogues in df_val['utterances']))

    # 更新训练和验证数据
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    for text, label in zip(df_train['utterances'], df_train['emotions']):
        combined_text, combined_label = combine_dialogues_and_labels(text, label, max_dialogue_length)
        train_texts.append(combined_text)
        train_labels.append(combined_label)

    for text, label in zip(df_val['utterances'], df_val['emotions']):
        combined_text, combined_label = combine_dialogues_and_labels(text, label, max_dialogue_length)
        val_texts.append(combined_text)
        val_labels.append(combined_label)