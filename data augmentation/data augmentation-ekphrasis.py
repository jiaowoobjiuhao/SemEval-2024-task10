import json
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# 路径到您的 JSON 文件
file_path = 'MaSaC_train_erc.json'

# 初始化 ekphrasis 文本处理器
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

def process_text(text):
    return " ".join(text_processor.pre_process_doc(text))

# 加载 JSON 数据
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理每个 utterance
for entry in data:
    entry['utterances'] = [process_text(utterance) for utterance in entry['utterances']]

# 保存修改后的数据
modified_file_path = 'Modified_MaSaC_train_erc.json'
with open(modified_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)



