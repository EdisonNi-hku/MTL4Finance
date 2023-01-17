from transformers import BertConfig, BertModel, BertTokenizerFast

print("Loading uncased BERT")
BertConfig.from_pretrained('bert-base-uncased', cache_dir='cache')
BertModel.from_pretrained('bert-base-uncased', cache_dir='cache')
BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir='cache')

print("Loading cased BERT")
BertConfig.from_pretrained('bert-base-cased', cache_dir='cache')
BertModel.from_pretrained('bert-base-cased', cache_dir='cache')
BertTokenizerFast.from_pretrained('bert-base-cased', cache_dir='cache')

print("Loading FinancialBERT")
BertConfig.from_pretrained('ahmedrachid/FinancialBERT', cache_dir='cache')
BertModel.from_pretrained('ahmedrachid/FinancialBERT', cache_dir='cache')
BertTokenizerFast.from_pretrained('ahmedrachid/FinancialBERT', cache_dir='cache')

print("Loading Y-FinBERT")
BertConfig.from_pretrained('yiyanghkust/finbert-pretrain', cache_dir='cache')
BertModel.from_pretrained('yiyanghkust/finbert-pretrain', cache_dir='cache')
BertTokenizerFast.from_pretrained('yiyanghkust/finbert-pretrain', cache_dir='cache')

print("Loading P-FinBERT")
# The P-FinBERT checkpoint on huggingface is finetuned on Financial PhraseBank already.
# Please download the pre-trained checkpoint at https://github.com/ProsusAI/finBERT
BertConfig.from_pretrained('ProsusAI/finbert', cache_dir='cache')
BertTokenizerFast.from_pretrained('ProsusAI/finbert', cache_dir='cache')

