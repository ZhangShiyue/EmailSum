import torch
from os import path
import numpy as np
import ujson as json
from tqdm import tqdm

from torch.utils.data import TensorDataset
from transformers import T5Tokenizer
import nltk
nltk.data.path.append('./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')


def process_file_t5(filename, tokenizer, lower=True):
    examples, eval = [], {}
    total = 0
    input_lens = []
    output_lens = []
    with open(f"{filename}.source", 'r') as fa, open(f"{filename}.target", 'r') as fs:
        while True:
            total += 1
            article = fa.readline()
            summary = fs.readline()
            if not article or not summary:
                break
            article = article.strip()
            summary = summary.strip()
            if lower:
                article = article.lower()
                summary = summary.lower()
            if not article or not summary:
                article = 'null'
                summary = 'null'
            inputs = article.split('|||')
            entities, input_tokens, speaker_tokens = {}, [], []
            turns = []
            # add prompt
            prompt = tokenizer.tokenize("summarize:")
            input_tokens.extend(prompt)
            turns.extend([0] * len(prompt))
            for i, input in enumerate(inputs):
                input = tokenizer.tokenize(input.strip())
                turns.extend([i + 1] * len(input))
                input_tokens.extend(input)
            # summary
            output_tokens = tokenizer.tokenize(summary)
            input_lens.append(len(input_tokens))
            output_lens.append(len(output_tokens))
            example = {"input": input_tokens, "output": output_tokens, "turn": turns, "id": total}
            eval[str(total)] = (article, summary)
            examples.append(example)
        np.random.shuffle(examples)
        print("{} examples in total".format(len(examples)))
        print("max_input: {}, max_output: {}.".format(max(input_lens), max(output_lens)))
        print("avg_input: {}, avg_output: {}.".format(np.mean(input_lens), np.mean(output_lens)))
    return examples, eval


def build_features_t5(examples, data_type, out_file, tokenizer, max_input_length, max_output_length):
    print("Processing {} examples...".format(data_type))
    total = 0
    input_inputs, output_inputs, turn_inputs, ids = [], [], [], []

    for example in tqdm(examples):
        total += 1

        # task input
        input_input = tokenizer.convert_tokens_to_ids(example["input"])[:max_input_length]
        input_inputs.append(input_input + [0] * (max_input_length - len(input_input)))

        # turn input
        turn_input = [turn for turn in example["turn"]][:max_input_length]
        turn_inputs.append(turn_input + [0] * (max_input_length - len(turn_input)))

        # task output
        output = tokenizer.convert_tokens_to_ids(example["output"])
        output_input = [0] + output[:max_output_length - 2] + [tokenizer.eos_token_id]
        output_inputs.append(output_input + [0] * (max_output_length - len(output_input)))

        ids.append(int(example["id"]))

    input_inputs = torch.tensor(input_inputs, dtype=torch.long)
    turn_inputs = torch.tensor(turn_inputs, dtype=torch.long)
    output_inputs = torch.tensor(output_inputs, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)

    dataset = TensorDataset(input_inputs, output_inputs, turn_inputs, ids)
    torch.save({"dataset": dataset}, out_file)
    print("Built {} instances of features in total".format(total))


def prepare_t5(tokenizer, data_dir, max_input_length, max_output_length, lower=True):
    train_file = f"{data_dir}/train"
    dev_file = f"{data_dir}/dev"
    test_file = f"{data_dir}/test"
    train_out = f"{data_dir}/train_{max_input_length}_{max_output_length}.t5"
    dev_out = f"{data_dir}/dev_{max_input_length}_{max_output_length}.t5"
    test_out = f"{data_dir}/test_{max_input_length}_{max_output_length}.t5"
    # process files
    if path.exists(train_file + '.source'):
        print(f"prepare {train_out}")
        train_examples, train_eval = process_file_t5(train_file, tokenizer, lower=lower)
        build_features_t5(train_examples, "train", train_out, tokenizer, max_input_length=max_input_length,
                          max_output_length=max_output_length)
    if path.exists(dev_file + '.source'):
        print(f"prepare {dev_out}")
        dev_examples, dev_eval = process_file_t5(dev_file, tokenizer, lower=lower)
        build_features_t5(dev_examples, "dev", dev_out, tokenizer, max_input_length=max_input_length,
                          max_output_length=max_output_length)
    if path.exists(test_file + '.source'):
        print(f"prepare {test_out}")
        test_examples, test_eval = process_file_t5(test_file, tokenizer, lower=lower)
        build_features_t5(test_examples, "test", test_out, tokenizer, max_input_length=max_input_length,
                          max_output_length=max_output_length)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the *.txt files for the task.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_input_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be discarded, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_output_length",
        default=56,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be discarded, sequences shorter will be padded.",
    )
    parser.add_argument("--lower", default=False, type=bool, help="Lower case")
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained("t5-base", cache_dir=args.cache_dir)
    prepare_t5(tokenizer, args.data_dir, args.max_input_length, args.max_output_length, args.lower)
