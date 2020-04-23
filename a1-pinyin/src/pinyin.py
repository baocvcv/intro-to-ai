import argparse
from console_progressbar import ProgressBar
import io
import time

from ngram import NGramModel
from ngram_zhuyin import NGramPYModel
from fenci_ngram import XNGramModel

parser = argparse.ArgumentParser(description='Pinyin input with N-gram.')
parser.add_argument('-f', '--fenci', dest='fenci', action='store_true',
                    help='N-gram on single character or phrase')
parser.add_argument('-z', '--no-zhuyin', dest='zhuyin', action='store_false',
                    help='To disable N-gram with zhuyin')
parser.add_argument('-i', '--input', dest='input', type=str,
                    metavar='FILE', help='Path to input pinyin file')
parser.add_argument('-o', '--output', dest='output', type=str,
                    metavar='FILE', help='Path to output file')
parser.add_argument('-s', '--source', dest='source', type=str, default='train',
                    metavar='FILEPATH', help='Path to training source file')
parser.add_argument('-m', '--model', dest='model', type=str, default='models/n-gram',
                    metavar='FILEPATH', help='Path to model files')
parser.add_argument('-n', dest='n', default=3, type=int,
                    metavar='NGRAM', help='Default as 3')
parser.add_argument('task', type=str, default='translate',
                    choices=['train', 'retrain', 'translate', 'test', 'console'],
                    help='Train, translate only, test accuracy, or use console mode')

def check_result(output: list, truth: list) -> float:
    correct_sentence_cnt = 0
    word_cnt = 0
    correct_word_cnt = 0
    for o, t in zip(output, truth):
        if o.strip() == t.strip():
            correct_sentence_cnt += 1
        word_cnt += len(o)
        for i in range(len(o.strip())):
            if o[i] == t[i]:
                correct_word_cnt += 1
    return (correct_sentence_cnt / len(output), correct_word_cnt / word_cnt)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.fenci:
        model = XNGramModel(
            n=args.n,
            table_path='pinyin_table',
            file_path=args.source,
            model_path=args.model,
            zhuyin=args.zhuyin)
    else:
        if args.zhuyin:
            model = NGramPYModel(
                n=args.n,
                table_path='pinyin_table',
                file_path=args.source,
                model_path=args.model)
        else:
            model = NGramModel(
                n=args.n,
                table_path='pinyin_table',
                file_path=args.source,
                model_path=args.model)

    if args.task == 'train':
        model.train([args.n-1])
    elif args.task == 'retrain':
        model.train(range(args.n))
    elif args.task == 'translate':
        if args.input is None:
            print('[Error] Missing input file.')
            exit(-1)
        model.load_model()
        lines = io.open(args.input, mode='r', encoding='utf-8').readlines()
        pb = ProgressBar(len(lines), length=50, prefix='Translating')
        result = []
        for i, l in enumerate(lines):
            result.append(model.translate(l))
            pb.print_progress_bar(i+1)
        print()
        print("[Info] Translated %d lines." % len(result))
        if args.output is None:
            for l in result:
                print(l)
        else:
            output = io.open(args.output, mode='w', encoding='utf-8')
            for l in result:
                output.write(l + '\n')
            print("[Info] Results saved to ", args.output)
    elif args.task == 'test':
        if args.input is None:
            print('[Error] Missing input file.')
            exit(-1)
        model.load_model()
        lines = io.open(args.input, mode='r', encoding='utf-8').readlines()
        pb = ProgressBar(len(lines) / 2, length=50, prefix='Translating')
        result = []
        for i, l in enumerate(lines[0::2]):
            result.append(model.translate(l))
            pb.print_progress_bar(i+1)
        print()
        if args.output is not None:
            output = io.open(args.output, mode='w', encoding='utf-8')
            for l in result:
                output.write(l + '\n')
            print("[Info] Results saved to ", args.output)
        accuracy = check_result(result, lines[1::2])
        print('[Info] Generated %d lines, with accuracy =' % len(result), accuracy)
    elif args.task == 'console':
        model.load_model()
        print("[Info] Entering console mode. Use Ctrl-C/D to exit.")
        while True:
            in_s = input(">> Input: ")
            time_d = time.time()
            result = model.translate(in_s)
            time_d = round(time.time()-time_d, 5)
            print(result)
            print("Used %fs" % time_d)
