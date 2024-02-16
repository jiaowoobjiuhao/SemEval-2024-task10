#调整学习率1e-5 然后用1000多个数据训练
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 模型1-deberta-base-focalloss-rdrop-类别平衡采样（任务1训练加预测一起代码）
import os
import sys
import logging
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, BertConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import WeightedRandomSampler
# from peft import PromptEncoderConfig, get_peft_model, TaskType, prepare_model_for_int8_training
from peft import AdaLoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW, DebertaConfig, AutoTokenizer
from transformers import DebertaV2ForSequenceClassification
from transformers import DebertaV2PreTrainedModel
from transformers import DebertaV2Model
from transformers import DebertaV2Tokenizer
from transformers import DebertaPreTrainedModel, DebertaModel
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW, DebertaConfig, AutoTokenizer
from transformers import DebertaForSequenceClassification
from transformers import DebertaPreTrainedModel
from transformers import DebertaModel
from transformers import DebertaTokenizer

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=0.3, weight=None, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights, ignore_index=-1)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1),
                    F.softmax(target, dim=-1), reduction=reduction)
    return loss

class DebertScratch(DebertaPreTrainedModel):
    def __init__(self, config, sep_token_id):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.sep_token_id = sep_token_id
        self.deberta = DebertaModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_labels)
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        batch_size, seq_len, _ = logits.size()
        all_utterance_logits = []
        for i in range(batch_size):
            sep_indices = (input_ids[i] == self.sep_token_id).nonzero(as_tuple=False).squeeze()
            start_idx = 0  # 开始于第一个[SEP]之后
            utterance_logits = []
             # 打印 sep_indices
            # print(f"sep_indices: {sep_indices}")
            for end_idx in sep_indices:
                # print(f"start_idx: {start_idx}, end_idx: {end_idx}")
                if start_idx >= end_idx:  # 避免空的话语
                    continue
                utterance_logit = logits[i, start_idx:end_idx].mean(dim=0)
                utterance_logits.append(utterance_logit)
                start_idx = end_idx + 1
            utterance_logits = torch.stack(utterance_logits)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            if utterance_logits.size(0) < labels.size(1):
                padding = torch.zeros(labels.size(1) - utterance_logits.size(0), self.num_labels, device=utterance_logits.device)
                utterance_logits = torch.cat([utterance_logits, padding], dim=0)
            all_utterance_logits.append(utterance_logits)

        all_utterance_logits = torch.stack(all_utterance_logits)

        total_loss = None
        if labels is not None:
            # 创建一个掩码，用于忽略填充的标签 (-1)
            mask = (labels != -1).unsqueeze(-1).expand(-1, -1, self.num_labels)
            valid_logits = all_utterance_logits[mask].view(-1, self.num_labels)  # 重塑为二维
            valid_labels = labels[mask[:, :, 0]].view(-1).long()  # 重塑为一维并确保数据类型为 long

            if valid_logits.size(0) == valid_labels.size(0):
                loss_fct = torch.nn.CrossEntropyLoss()
                ce_loss = loss_fct(valid_logits, valid_labels)

                kl_loss = (KL(valid_logits, valid_logits) + KL(valid_logits, valid_logits)) / 2
                total_loss = ce_loss + kl_loss
            else:
                raise RuntimeError("Logits and labels shape mismatch after mask")
        if return_dict:
            return SequenceClassifierOutput(
                loss=total_loss,
                logits=all_utterance_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
        else:
            return (total_loss, all_utterance_logits)
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, num_labels):
        self.encodings = encodings
        self.labels = labels
        self.num_labels = num_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item_labels = torch.tensor(self.labels[idx], dtype=torch.long)

        # 打印调试信息
        # print(f"Sample index: {idx}")
        # print(f"Encoded input: {item['input_ids']}")
        # print(f"Labels: {item_labels}")

        # 检查标签是否在合理范围内，忽略-1填充值
        assert ((item_labels >= 0) & (item_labels < self.num_labels) | (item_labels == -1)).all(), \
            "标签值必须在 0 到 num_classes-1 的范围内或为 -1"

        item['labels'] = item_labels
        return item
    def __len__(self):
        return len(self.labels)

    def get_class_counts(self):
        # 为每个可能的标签创建一个计数字典，并将计数初始化为0
        class_counts = {i: 0 for i in range(self.num_labels)}
        for labels in self.labels:
            for label in labels:
                if label != -1:  # 忽略填充值
                    class_counts[label] += 1
        return class_counts
# 这个方法返回了数据集中样本的总数，通常是 labels 列表的长度，以确保数据集对象知道有多少个样本可供处理
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    # 加载数据
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

    path = "microsoft/deberta-base"
    # 确保你的 tokenizer 和 encodings 步骤是正确的
    tokenizer = DebertaTokenizer.from_pretrained(path)
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=2048)
    val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=2048)
    num_labels = len(unique_labels)  # 根据实际情况设置标签数量
    train_dataset = TrainDataset(train_encodings, train_labels, num_labels)
    val_dataset = TrainDataset(val_encodings, val_labels, num_labels)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_labels = len(unique_labels)  # 根据实际情况设置标签数量
    config = DebertaConfig.from_pretrained(path, max_position_embeddings=2048)
    # 根据需要修改分类器的维度（根据 num_labels 参数）
    config.num_labels = num_labels  # 根据实际情况设置标签数量
    model = DebertScratch(config=config,sep_token_id=tokenizer.sep_token_id)
    model.to(device)
    model.train()
    class_counts = train_dataset.get_class_counts()
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (class_counts[i] if class_counts[i] > 0 else 1) for i in range(num_labels)]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 为 FocalLoss 和 WeightedRandomSampler 使用同一组类别权重
    criterion = FocalLoss(alpha=0.1, gamma=0.3, reduction='mean', class_weights=class_weights_tensor)

    # 为数据集创建 WeightedRandomSampler
    sampler_weights = 1.0 / torch.tensor(class_weights, dtype=torch.float)
    # 对类别权重进行调整
    adjusted_class_weights = [w * 0.5 for w in class_weights]
    sampler = WeightedRandomSampler(weights=adjusted_class_weights, num_samples=len(train_dataset), replacement=True)

    # 为数据集创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    best_val_f1 = 0.0
    #best_val_loss = float('inf')
    for epoch in range(5):
        start = time.time()
        train_loss, val_loss = 0, 0
        train_f1, val_f1 = 0, 0
        train_precision, val_precision = 0, 0
        train_recall, val_recall = 0, 0
        n, m = 0, 0
        #criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
        with tqdm(total=len(train_loader), desc="Epoch %d" % epoch) as pbar:
            for batch in train_loader:
                n += 1
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                labels = batch['labels'].to(device)
                labels = labels.view(-1).long()  # 确保标签为长整型
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                # 修改损失计算逻辑以适应新的输出格式
                loss = criterion(logits.view(-1, num_labels), labels)

                # 将 logits 转换为预测的类别标签
                predictions = torch.argmax(logits, dim=-1).view(-1)
                # 重新获取未重塑的 labels 用于评估
                labels_eval = batch['labels'].view(-1).cpu().numpy()

                # 计算 F1, precision 和 recall
                train_f1 += f1_score(labels_eval, predictions.cpu().numpy(), average='micro')
                train_precision += precision_score(labels_eval, predictions.cpu().numpy(), average='micro')
                train_recall += recall_score(labels_eval, predictions.cpu().numpy(), average='micro')

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.005)  # 梯度裁剪
                optimizer.step()
                train_loss += loss.item()  # 使用 .item() 获取数值
                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss/ n),
                                  'train F1': '%.2f' % (train_f1 / n),
                                  'train Precision': '%.2f' % (train_precision / n),
                                  'train Recall': '%.2f' % (train_recall / n)})

                pbar.update(1)
        # Validation Loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                m += 1
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                # 创建一个掩码，用于忽略填充的标签 (-1)
                mask = (labels != -1)
                valid_logits = logits[mask].view(-1, num_labels)
                valid_labels = labels[mask].view(-1).long()

                if valid_logits.size(0) == valid_labels.size(0):
                    loss = criterion(valid_logits, valid_labels)
                    val_loss += loss.item()

                    # 将 logits 转换为预测的类别标签
                    predictions = torch.argmax(valid_logits, dim=-1).view(-1)
                    valid_labels = valid_labels.cpu().numpy()

                    # 计算 F1, precision 和 recall
                    val_f1 += f1_score(valid_labels, predictions.cpu().numpy(), average='micro')
                    val_precision += precision_score(valid_labels, predictions.cpu().numpy(), average='micro')
                    val_recall += recall_score(valid_labels, predictions.cpu().numpy(), average='micro')
                else:
                    raise RuntimeError("Logits and labels shape mismatch after mask")
        end = time.time()
        runtime = end - start
        pbar.set_postfix({'epoch': '%d' % (epoch),
                          'train loss': '%.4f' % (train_loss / n),
                          'train F1': '%.2f' % (train_f1 / n),
                          'train Precision': '%.2f' % (train_precision / n),
                          'train Recall': '%.2f' % (train_recall / n),
                          'val loss': '%.4f' % (val_loss / m),
                          'val F1': '%.2f' % (val_f1 / m),
                          'val Precision': '%.2f' % (val_precision / m),
                          'val Recall': '%.2f' % (val_recall / m),
                          'time': '%.2f' % (runtime)})
        current_val_f1 = val_f1 / m
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                # 可以添加其他状态信息
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
            }, 'best_model_state_dict.pth')

            print(f"Epoch {epoch}: New best val F1: {best_val_f1:.4f}. Model saved.")
        model.train()  # 继续回到第二轮循环
        logging.info('Validation Metrics:')
        logging.info('val loss: %.4f' % (val_loss/ m))
        logging.info('val F1: %.2f' % (val_f1 / m))
        logging.info('val Precision: %.2f' % (val_precision / m))
        logging.info('val Recall: %.2f' % (val_recall / m))
        logging.info('result saved!')
        print("Training completed.")







#预测代码
#deberta-base 预测
import pandas as pd
import torch
from transformers import DebertaPreTrainedModel,DebertaModel,DebertaTokenizer,DebertaConfig
from torch.utils.data import DataLoader
from peft import AdaLoraConfig, get_peft_model
class DebertScratch(DebertaPreTrainedModel):
    def __init__(self, config, sep_token_id):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.sep_token_id = sep_token_id
        self.deberta =DebertaModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_labels)
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        batch_size, seq_len, _ = logits.size()
        all_utterance_logits = []

        for i in range(batch_size):
            sep_indices = (input_ids[i] == self.sep_token_id).nonzero(as_tuple=False).squeeze()
            start_idx = 1
            utterance_logits = []
            for end_idx in sep_indices:
                if start_idx >= end_idx:
                    continue
                utterance_logit = logits[i, start_idx:end_idx].mean(dim=0)
                utterance_logits.append(utterance_logit)
                start_idx = end_idx + 1
            utterance_logits = torch.stack(utterance_logits)
            if labels is not None:
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                if utterance_logits.size(0) < labels.size(1):
                    padding = torch.zeros(labels.size(1) - utterance_logits.size(0), self.num_labels, device=utterance_logits.device)
                    utterance_logits = torch.cat([utterance_logits, padding], dim=0)
            all_utterance_logits.append(utterance_logits)

        all_utterance_logits = torch.stack(all_utterance_logits)

        total_loss = None
        if labels is not None:
            mask = (labels != -1).unsqueeze(-1).expand(-1, -1, self.num_labels)
            valid_logits = all_utterance_logits[mask].view(-1, self.num_labels)
            valid_labels = labels[mask[:, :, 0]].view(-1).long()

            if valid_logits.size(0) == valid_labels.size(0):
                loss_fct = torch.nn.CrossEntropyLoss()
                ce_loss = loss_fct(valid_logits, valid_labels)

                kl_loss = (KL(valid_logits, valid_logits) + KL(valid_logits, valid_logits)) / 2
                total_loss = ce_loss + kl_loss
            else:
                raise RuntimeError("Logits and labels shape mismatch after mask")
        if return_dict:
            return SequenceClassifierOutput(
                logits=all_utterance_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
        else:
            return (all_utterance_logits,)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# 加载测试数据
df_test = pd.read_json('Modified_MaSaC_test_erc.json')
test_texts = [" [SEP] ".join(dialogue) for dialogue in df_test['utterances']]

# 加载tokenizer
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=2048)

# 创建测试数据集
test_dataset = TestDataset(test_encodings)

# 加载模型
model_path = 'best_model_state_dict.pth'
config = DebertaConfig.from_pretrained('microsoft/deberta-base', num_labels=8,max_position_embeddings=2048)
model = DebertScratch(config=config,sep_token_id=tokenizer.sep_token_id)

# 加载模型权重
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 创建 ID 到标签的映射
int_to_emotion = {idx: emotion for emotion, idx in emotion_to_int.items()}

# 进行预测
predictions = []
with torch.no_grad():
    for batch in DataLoader(test_dataset, batch_size=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits.squeeze(0)  # 移除批次维度

        dialogue_emotions = []
        sep_indices = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=False).squeeze()
        start_idx = 0
        for end_idx in sep_indices:
            if start_idx >= end_idx:  # 跳过空话语
                continue
            # 找到对应话语的最大预测情绪
            utterance_logit = logits[start_idx:end_idx].mean(dim=0)
            predicted_label_idx = torch.argmax(utterance_logit).item()
            predicted_emotion = int_to_emotion[predicted_label_idx]
            dialogue_emotions.append(predicted_emotion)
            start_idx = end_idx + 1

        predictions.append(dialogue_emotions)

# 保存预测结果
# 保存预测结果到 'answer.txt'
with open('answer.txt', 'w', encoding='utf-8') as file:
    for emotions in predictions:
        for emotion in emotions:
            file.write(f"{emotion.lower()}\n")
print("预测结果已保存至 prediction_results.txt")