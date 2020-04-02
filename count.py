from zhon.hanzi import punctuation as cp
from string import punctuation as ep
import re
import json
import io

punc = re.compile(r"[%s%s0-9]+" % (cp, ep))
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
