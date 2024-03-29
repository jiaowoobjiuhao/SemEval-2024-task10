class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# def KL(input, target, reduction="sum"):
#     input = input.float()
#     target = target.float()
#     loss = F.kl_div(F.log_softmax(input, dim=-1),
#                     F.softmax(target, dim=-1), reduction=reduction)
#     return loss


class DebertScratch(DebertaV2PreTrainedModel):
    def __init__(self, config, sep_token_id):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.sep_token_id = sep_token_id
        self.deberta = DebertaV2Model(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
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
                # 确保 padding 张量的第二个维度大小与 utterance_logits 的第二个维度大小一致
                padding = torch.zeros(labels.size(1) - utterance_logits.size(0), utterance_logits.size(-1), device=utterance_logits.device)
                utterance_logits = torch.cat([utterance_logits, padding], dim=0)

            all_utterance_logits.append(utterance_logits)

        all_utterance_logits = torch.stack(all_utterance_logits)

        total_loss = None
        if labels is not None:
            mask = (labels != -1.0).unsqueeze(-1).expand(-1, -1, 1)
            valid_logits = all_utterance_logits[mask].squeeze(-1)  # 调整为一维，因为只有一个输出
            valid_labels = labels[mask[:, :, 0]].view(-1).float()  # 调整为一维

            if valid_logits.size(0) == valid_labels.size(0):
                loss_fct = torch.nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss
                total_loss = loss_fct(valid_logits, valid_labels)
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
        class_counts = {0: 0, 1: 0}
        for labels in self.labels:
            for label in labels:
                if label != -1:  # 忽略填充值
                    class_counts[label] += 1
        return class_counts