import argparse
from src.ngram import NGramModel
from src.ngram_zhuyin import NGramPYModel

parser = argparse.ArgumentParser(description='Pinyin input with N-gram.')
parser.add_argument('-f', '--fenci', dest='fenci', action='store_true',
                    help='N-gram on single character or phrase')
parser.add_argument('-z', '--zhuyin', dest='zhuyin', action='store_true',
                    help='To enable N-gram with zhuyin')
parser.add_argument('-i', '--input', dest='input', type=argparse.FileType('r'),
                    metavar='FILE', help='Path to input pinyin file')
parser.add_argument('-o', '--output', dest='output', type=argparse.FileType('w'),
                    metavar='FILE', help='Path to output file')
parser.add_argument('-s', '--source', dest='source', type=str, default='train',
                    metavar='FILEPATH', help='Path to training source file')
parser.add_argument('-m', '--model', dest='model', type=str, default='src/models/n-gram',
                    metavar='FILEPATH', help='Path to model files')
parser.add_argument('-n', dest='n', default=2, type=int,
                    metavar='NGRAM', help='Default as 2')
parser.add_argument('task', type=str,
                    choices=['train', 'retrain', 'translate', 'test', 'console'],
                    help='Train, translate only, test accuracy, or use console mode')

def check_result(output: list, truth: list) -> float:
    cnt = 0
    for o, t in zip(output, truth):
        if o.strip() == t.strip():
            cnt += 1
    return cnt * 1.0 / len(output)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.zhuyin:
        model = NGramPYModel(
            n=args.n,
            table_path='pinyin_table',
            file_path=args.source,
            model_path=args.model)
    elif args.fenci:
        model = None
    else:
        model = NGramModel(
            n=args.n,
            table_path='pinyin_table',
            file_path=args.source,
            model_path=args.model)

    if args.task == 'train':
        model.train()
    elif args.task == 'retrain':
        model.train(force=True)
    elif args.task == 'translate':
        if args.input is None:
            print('[Error] Missing input file.')
            exit(-1)
        model.load_model()
        result = [model.translate(l) for l in args.input.readlines()]
        print("[Info] Translated %d lines." % len(result))
        if args.output is None:
            print(result)
        else:
            args.output.writelines(result)
            print("[Info] Results saved to ", args.output.name)
    elif args.task == 'test':
        if args.input is None:
            print('[Error] Missing input file.')
            exit(-1)
        model.load_model()
        lines = args.input.readlines()
        result = [model.translate(l) for l in lines[0::2]]
        print("[Info] Results:")
        if args.output is None:
            print(result)
        else:
            args.output.writelines(result)
            print("[Info] Results saved to ", args.output.name)
        accuracy = check_result(result, lines[1::2])
        print('[Info] Generated %d lines, with accuracy = %f' % (len(result), accuracy))
    elif args.task == 'console':
        model.load_model()
        print("[Info] Entering console mode. Use Ctrl-C/D to exit.")
        while True:
            in_s = input(">> Input: ")
            result = model.translate(in_s)
            print(result)
