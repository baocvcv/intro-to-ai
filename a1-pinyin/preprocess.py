import re
import os
import io
import sys
import json
import multiprocessing

from pypinyin import lazy_pinyin
from jieba import lcut

def process(fin: str, fout: str):
    print('Processing', fin)
    punc = re.compile(r"[^\u4e00-\u9fff]+")
    file_content = io.open(fin,
                           mode='r',
                           encoding='utf-8').readlines()
    fout = io.open(fout, mode='w', encoding='utf-8')
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
    print('Saved to', fout)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py [Folder_in] [Folder_out]")
        exit(-1)

    folder_in = sys.argv[1]
    folder_out = sys.argv[2]
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    
    args = []
    for fname in os.listdir(folder_in):
        if 'readme' in fname.lower():
            continue
        args.append((os.path.join(folder_in, fname),
                    os.path.join(folder_out, fname)))
    with multiprocessing.Pool(8) as pool:
        pool.starmap(process, args)
        # process(os.path.join(folder_in, fname),
        #         os.path.join(folder_out, fname))