import argparse
import torch
from utils.helpers import read_lines
from gector.custom_gec_model import GecBERTModel
import time
import numpy as np
import GPUtil


def predict_for_file(input_file, output_file, model, batch_size=32):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    rec_sent_times = []; rec_word_times = []
    for sent in test_data:
        # print('sent:', sent)
        batch.append(sent.split())
        # print('batch:', batch); exit(0)
        if len(batch) == batch_size:
            # print('batch:', batch)
            # print('batch:', len(batch))
            tot_words = sum([len(item) for item in batch])
            # print('tot_words:', tot_words)
            # print('before')
            # GPUtil.showUtilization()

            start = time.time()
            preds, cnt = model.handle_batch(batch)
            stop = time.time()
            tot_time = stop-start

            # print('after')
            # GPUtil.showUtilization()
            # print('after empty:')
            torch.cuda.empty_cache()
            # GPUtil.showUtilization()
            # exit(0)

            # print('time taken:', tot_time); print('sent/sec:', len(batch)/tot_time, 'words/sec:', tot_words/tot_time);
            rec_sent_times.append(len(batch)/tot_time); rec_word_times.append(tot_words/tot_time)
            # exit(0)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        start = time.time()
        preds, cnt = model.handle_batch(batch)
        stop=time.time()
        tot_words = sum([len(item) for item in batch])
        tot_time = stop - start
        rec_sent_times.append(len(batch) / tot_time); rec_word_times.append(tot_words / tot_time)

        predictions.extend(preds)
        cnt_corrections += cnt

    # print('first batch:', rec_sent_times[0], rec_word_times[0])
    print('Mean sent/sec:', np.mean(rec_sent_times), 'Mean words/sec:', np.mean(rec_word_times))
    print('Median sent/sec:', np.median(rec_sent_times), 'Median words/sec:', np.median(rec_word_times))

    with open(output_file, 'w') as f:
        f.write("\n".join([" ".join(x) for x in predictions]) + '\n')
    return cnt_corrections


def main(args):
    # get all paths
    # print('chk1')
    # GPUtil.showUtilization()
    # exit(0)
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         min_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights,
                         prune_amount=args.prune_amount
                         )
    # GPUtil.showUtilization()
    # print('chk2')
    # # exit(0)

    cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'distilbert', 'gpt2', 'roberta', 'transformerxl', 'xlnet', 'albert', 'distilgpt2'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--prune_amount',
                        type=float,
                        help='l1 Unstructured prune amount',
                        default=0.)
    args = parser.parse_args()
    main(args)
