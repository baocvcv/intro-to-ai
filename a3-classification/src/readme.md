## 代码说明

### main.py：训练、测试、调参

- `python3 main.py -h` 查看帮助

  ```
  usage: main.py [-h] --model {tCNN,tRNN,dpCNN,tRNN_att,mlp}
                 {train,test,tune,log}
  
  Text Classification
  
  positional arguments:
    {train,test,tune,log}
                          train or test the model
  
  optional arguments:
    -h, --help            show this help message and exit
    --model {tCNN,tRNN,dpCNN,tRNN_att,mlp}, -m {tCNN,tRNN,dpCNN,tRNN_att,mlp}
                          choose a model
  ```

- `python3 -m <model_name> train` 来训练模型
- 其余功能用来调试参数



### util.py：数据处理

- `python3 util.py -h` 查看帮助

```
usage: util.py [-h] [--input INPUT] [--vocab VOCAB] [--embedding EMBEDDING]
               [--output OUTPUT] [--ratio RATIO]
               {split,check,build,boost}

Parse training data with word vectors

positional arguments:
  {split,check,build,boost}

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        training data file
  --vocab VOCAB, -v VOCAB
                        vocabulary dictionary
  --embedding EMBEDDING, -e EMBEDDING
                        word vector file
  --output OUTPUT, -o OUTPUT
                        output file
  --ratio RATIO, -r RATIO
                        ratio of valid
```

- `python3 util.py split`用来划分训练集和验证集
- `python3 util.py check`用来统计训练数据的分布
- `python3 util.py build`使用词向量的源文件，来生成规模较小的词向量用于训练



### train_eval.py：训练模型和测试的帮助函数

### text_models/base_config.py：模型训练的基本设置

### text_models/其他文件：各个模型的具体设置和代码实现

