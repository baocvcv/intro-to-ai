from src.ngram import NGramModel

model = NGramModel(
    n=2,
    table_path='pinyin_table',
    file_path='test_news')
# model.train()
# model.save_model(toDir='src/models/3-gram')
model.load_model(fromDir='src/models/2-gram')
model.translate("xin hua she")
