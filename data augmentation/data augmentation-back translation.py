import pandas as pd
import translators as ts
import langdetect  # 用于语言检测

# 加载JSON文件
file_path = 'Modified_MaSaC_train_erc.json'
df = pd.read_json(file_path)

# 定义翻译器和翻译函数
API = 'baidu'
lang_dict = {'Hindi': 'hin', 'English': 'en'}

def translator_constructor(api):
    if api == 'google':
        return ts.google
    elif api == 'bing':
        return ts.bing
    elif api == 'baidu':
        return ts.baidu
    # ... 其他翻译服务 ...

def back_translate(text):
    try:
        # 检测文本语言
        detected_language = langdetect.detect(text)
        if detected_language == 'en':
            # 英语先翻译成印地语，再翻译回英语
            intermediate_translation = translator_constructor(API)(text, lang_dict['English'], lang_dict['Hindi'])
            return translator_constructor(API)(intermediate_translation, lang_dict['Hindi'], lang_dict['English'])
        else:
            # 印地语先翻译成英语，再翻译回印地语
            intermediate_translation = translator_constructor(API)(text, lang_dict['Hindi'], lang_dict['English'])
            return translator_constructor(API)(intermediate_translation, lang_dict['English'], lang_dict['Hindi'])
    except Exception as e:
        return text  # 出错时返回原始文本

# 创建一个新的 DataFrame 来存储回译后的数据
back_translated_df = df.copy()
back_translated_df['utterances'] = back_translated_df['utterances'].apply(lambda conv: [back_translate(utt) for utt in conv])

# 将回译后的数据追加到原始数据中
enhanced_df = pd.concat([df, back_translated_df])

# 保存处理后的文件
output_file_path = 'ekphrasis_MaSaC_train_erc_back_translated.json'
enhanced_df.to_json(output_file_path, orient='records', lines=True)
