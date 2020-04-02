import re
import json
import io

from pypinyin import lazy_pinyin

punc = re.compile(r"[^\u4e00-\u9fff]+")
lines = io.open('a1-pinyin/test_news/2016-11.txt',
                mode='r',
                encoding='utf-8').readlines()
content = [punc.sub(' ', json.loads(line)['html']) for line in lines]

cnt = 0
for line in content:
    cnt += line.count('新')
print(cnt)

cnt = 0
for line in content:
    cnt += line.count('新华')
print(cnt)

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
