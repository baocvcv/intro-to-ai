import re
import json
import io
from collections import Counter
import os

from pypinyin import lazy_pinyin

punc = re.compile(r"[^\u4e00-\u9fff]+")
cnt = Counter()
for fn in os.listdir('a1-pinyin/test_news/'):
    lines = io.open('a1-pinyin/test_news/' + fn,
                    mode='r',
                    encoding='utf-8').readlines()
    content = [punc.sub(' ', json.loads(line)['html']) for line in lines]

    for line in content:
        cnt['薮'] += line.count('薮')
        cnt['搜'] += line.count('搜')
        cnt['搜索'] += line.count('搜索')
    '''
    cnt = 0
    for line in content:
        py = lazy_pinyin(line)
        for i,w in enumerate(line):
            if w == '新':
                if py[i] == 'xin':
                    cnt += 1
                else:
                    print(line, i, py[i])
    print(cnt)
    '''
print(cnt)
