import re
import os
import io
import sys
import json

from pypinyin import lazy_pinyin
from jieba import lcut

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py [Folder]")
        exit(-1)

    folder = sys.argv[1]
    punc = re.compile(r"[^\u4e00-\u9fff]+")
    for fname in os.listdir(folder):
        if 'readme' in fname.lower():
            continue
        file_content = io.open(os.path.join(folder, fname),
                                mode='r',
                                encoding='utf-8').readlines()
        fout = io.open(os.path.join(folder, 'data-'+fname),
                            mode='w', encoding='utf-8')
        for line in file_content:
            line_content = punc.sub(' ', json.loads(line)['html'])
            for l in line_content.split():
                if l != '':
                    out = {'text': l,
                           'py': lazy_pinyin(l),
                           'fc': lcut(l)}
                    json.dump(out, fout)
                    fout.write('\n')
        fout.close()

