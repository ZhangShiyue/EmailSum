# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from metrics import rouge, bertScore, extScore

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    T5Config,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule
)
# from transformers.modeling_t5 import T5ForConditionalGeneration
from transformers.tokenization_utils import trim_batch
from utils import SortishSampler, get_inverse_sqrt_schedule_with_warmup, Adafactor
from models import T5ForConditionalGenerationHT5

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "t5": (T5Config, T5ForConditionalGenerationHT5, T5Tokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def prepare_batch(args, batch):
    input_ids = batch[0]
    decoder_input_ids = batch[1]
    if args.trim_batch:
        input_ids = trim_batch(input_ids, 0)
        decoder_input_ids = trim_batch(decoder_input_ids, 0)
        decoder_input_ids = torch.cat([torch.zeros(decoder_input_ids.size()[0], 1, dtype=torch.long).cuda(),
                                       decoder_input_ids], dim=1)

    if args.no_start_pad:
        decoder_input_ids = decoder_input_ids[:, 1:]
    lm_labels = decoder_input_ids[:, 1:].clone()
    lm_labels[decoder_input_ids[:, 1:] == 0] = -100
    decoder_input_ids = decoder_input_ids[:, :-1].contiguous()
    return input_ids, decoder_input_ids, lm_labels


def train(args, train_dataset, model, tokenizer, best_ckpt):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir, flush_secs=30)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
    else:
       if args.sampler == "sortish":
            train_sampler = SortishSampler(train_dataset, batch_size=args.train_batch_size)
       else:
            train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    save_steps = args.save_steps if args.save_steps \
        else len(train_dataloader) // args.gradient_accumulation_steps
    logging_steps = save_steps // 2

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.optimizer == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, relative_step=False)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.scheduler == "linear":
        warmup_steps = int(save_steps * args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    elif args.scheduler == "inverse_sqrt":
        warmup_steps = int(save_steps * args.warmup_steps)
        scheduler = get_inverse_sqrt_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = get_constant_schedule(optimizer)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            # 0 input_inputs, output_inputs, output_labels, ids
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, decoder_input_ids, lm_labels = prepare_batch(args, batch)
            attention_mask = input_ids.bool().long()
            batch_size, _ = input_ids.size()

            if args.memory_type == "ht5":
                input_ids = torch.cat([input_ids, batch[2]], dim=1)
                mem_attn_mask = torch.ones((batch_size, args.max_turn_length), dtype=torch.long).cuda()
                attention_mask = torch.cat([mem_attn_mask, attention_mask], dim=1)

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask,
                      "decoder_input_ids": decoder_input_ids, "lm_labels": lm_labels}
            dec_outputs = model(**inputs)
            loss = dec_outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if not args.zero_shot:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                        logger.info(f"step{global_step} {key}: {value}")
                    tb_writer.flush()

                if args.local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # evaluate
                    if (args.local_rank == -1 and args.evaluate_during_training):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, args.dev_data_file, args.dev_eval_file,
                                           args.dev_source_file, tokenizer,
                                           result_file=f"{output_dir}/dev.res", score_file=f"{output_dir}/dev.score",
                                           global_step=global_step, prefix=output_dir)
                        # update best results
                        for key in results:
                            if results[key] > best_ckpt[key][0]:
                                best_ckpt[key] = (results[key], os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))

                                # overwrite to key dirs
                                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(key))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)

                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                logger.info("Saving model checkpoint to %s", output_dir)

                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                        for key, value in results.items():
                            tb_writer.add_scalar(f"eval_{key}", value, global_step)
                        tb_writer.flush()

                        logger.info(f"step{global_step} write best checkpoint...")
                        with open(f"{args.output_dir}/best_ckpt.json", 'w') as f:
                            json.dump(best_ckpt, f)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_ckpt


def evaluate(args, model, dev_data_file, dev_eval_file, dev_source_file, tokenizer,
             result_file, score_file, global_step=-1, prefix="", test=False, dev_eval_file1=None):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir, flush_secs=30)

    eval_output_dir = args.output_dir

    eval_dataset = torch.load(dev_data_file)
    eval_dataset = eval_dataset["dataset"]

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if not hasattr(model, 'module') and args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    res = []
    dev_loss, count = 0., 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, decoder_input_ids, lm_labels = prepare_batch(args, batch)
            batch_size, _ = input_ids.size()
            ids = batch[-1].cpu().numpy()
            # 0 task_input_inputs, output_inputs, output_labels, ids
            model_to_generate = model.module if hasattr(model, 'module') else model
            attention_mask = input_ids.bool().long()
            batch_size, _ = input_ids.size()

            if args.memory_type == "ht5":
                input_ids = torch.cat([input_ids, batch[2]], dim=1)
                mem_attn_mask = torch.ones((batch_size, args.max_turn_length), dtype=torch.long).cuda()
                attention_mask = torch.cat([mem_attn_mask, attention_mask], dim=1)
            # dev loss
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask,
                      "decoder_input_ids": decoder_input_ids, "lm_labels": lm_labels}
            dec_outputs = model(**inputs)
            loss = dec_outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            dev_loss += loss.item()
            count += 1

            preds = model_to_generate.generate(input_ids=input_ids, attention_mask=attention_mask,
                                               max_length=args.max_output_length, min_length=args.min_output_length,
                                               length_penalty=args.length_penalty,
                                               no_repeat_ngram_size=args.no_repeat_ngram_size, early_stopping=True,
                                               do_sample=False, use_cache=True, num_beams=args.beam_size if test else 1)
            res.extend(detok(ids, preds.cpu().numpy(), batch_size, tokenizer))

    logs = {"dev_loss": dev_loss / (count + 1e-12)}
    for key, value in logs.items():
        logger.info(f"{key}: {value}")
        if global_step >= 0:
            tb_writer.add_scalar(key, value, global_step)

    result = score(args, res, dev_eval_file, dev_source_file, result_file,
                   score_file, ref_file1=dev_eval_file1)

    if test:
        output_eval_file = os.path.join(args.output_dir, prefix, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    else:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result


def detok(ids, preds, batch_size, tokenizer, separate=False):
    res = []
    for i in range(batch_size):
        pred = tokenizer.decode(preds[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        res.append((ids[i], pred))
    return res


def score(args, res, ref_file, source_file, result_file, score_file, ref_file1=None):
    res = sorted(res, key=lambda x: int(x[0]))
    res = [pred for id, pred in res]
    logger.info(f"***** Write generated summaries to {result_file} *****")
    with open(result_file, 'w') as f:
        f.write('\n'.join(res))
    logger.info(f"***** Write scores to {score_file if not ref_file1 else f'{score_file}.2ref'} *****")
    final_score = {}
    for refno, ref_file in enumerate([ref_file, ref_file1] if ref_file1 else [ref_file]):
        ref = open(ref_file, 'r').readlines()
        source = open(source_file, 'r').readlines()
        assert len(res) == len(ref) == len(source)
        score, score_str = rouge(ref, res)
        bscore, bscore_str = bertScore(ref, res)
        escore, escore_str = extScore(source, res)
        score.update(bscore)
        score.update(escore)
        for key in score:
            if key not in final_score:
                final_score[key] = []
            final_score[key].append(score[key])
    for key in final_score:
        final_score[key] = np.mean(final_score[key])  # take average
    res_str = '\n'.join(["%s = %.2f" %
                         (key, final_score[key]) for key in final_score])
    with open(f"{score_file}.2ref" if ref_file1 else score_file, 'w') as f:
        f.write(f"{res_str}")
    return final_score


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help="The dataset file used for training",
    )
    parser.add_argument(
        "--eval_file",
        default=None,
        type=str,
        help="The dataset file used for training",
    )
    parser.add_argument(
        "--source_file",
        default=None,
        type=str,
        help="The dataset file used for training",
    )
    parser.add_argument(
        "--dev_data_file",
        default=None,
        type=str,
        help="The dataset file used for evaluation",
    )
    parser.add_argument(
        "--dev_eval_file",
        default=None,
        type=str,
        help="The dataset file used for evaluation",
    )
    parser.add_argument(
        "--dev_source_file",
        default=None,
        type=str,
        help="The dataset file used for evaluation",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="The dataset file used for testing",
    )
    parser.add_argument(
        "--test_eval_file",
        default=None,
        type=str,
        help="The dataset file used for testing",
    )
    parser.add_argument(
        "--test_eval_file1",
        default=None,
        type=str,
        help="The dataset file used for testing",
    )
    parser.add_argument(
        "--test_source_file",
        default=None,
        type=str,
        help="The dataset file used for testing",
    )
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        help="The dataset file used for training evaluation",
    )
    parser.add_argument(
        "--train_eval_file",
        default=None,
        type=str,
        help="The dataset file used for training evaluation",
    )
    parser.add_argument(
        "--train_source_file",
        default=None,
        type=str,
        help="The dataset file used for training evaluation",
    )
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_input_length", default=512, type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_output_length", default=200, type=int,
        help="The maximum total output sequence length after tokenization.",
    )
    parser.add_argument(
        "--min_output_length", default=30, type=int,
        help="The maximum total output sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_turn_length", default=10, type=int,
        help="The maximum total turns of input.",
    )
    parser.add_argument("--length_penalty", default=2.0, type=float, help="Length penalty used for decoding.")
    parser.add_argument("--no_repeat_ngram_size", default=3, type=int, help="No repeat ngram size for decoding.")
    parser.add_argument("--beam_size", default=1, type=int, help="Beam size for decoding.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--zero_shot", action="store_true", help="Whether to update parameter.")
    parser.add_argument("--no_start_pad", action="store_true", help="Not start with pad token")
    parser.add_argument("--trim_batch", action="store_true", help="Trim batch")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--test_res", default="test", type=str, help="The name of test results")
    parser.add_argument("--dev_res", default="dev", type=str, help="The name of dev results")
    parser.add_argument(
        "--test_checkpoint",
        default=None,
        type=str,
        help="Which checkpoint to test",
    )
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--scheduler", default=None, type=str, help="learning rate scheduler: linear, inverse_sqrt")
    parser.add_argument("--optimizer", default="adam", type=str, help="The optimizer to use")
    parser.add_argument("--sampler", default=None, type=str, help="Dataset sampler: random, sortish")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0., type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=None, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--memory_type", default="base", type=str, help="The type of model")
    parser.add_argument("--from_pretrain", action="store_true", help="Load from pretrained model")
    parser.add_argument("--pretrained_checkpoint", default=None, type=str,
                        help="Load from pretrained model")
    parser.add_argument("--from_scratch", action="store_true", help="Train model from scratch")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # update config
    config.update({"max_input_length": args.max_input_length,
                   "memory_length": args.max_turn_length,
                   "memory_type": args.memory_type,
                   "dec_memory_length": args.max_turn_length,
                   "output_attentions": False})
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        do_lower_case=args.do_lower_case,
    )
    if args.from_scratch:
        model = model_class(config)
        model.init_weights()
    else:
        model = model_class.from_pretrained(
            args.pretrained_checkpoint if args.from_pretrain else args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    best_ckpt = {"rouge1": (0., 0),
                 "rouge2": (0., 0),
                 "rougeL": (0., 0),
                 "rougeLsum": (0., 0),
                 "ext_rouge2_prec": (0., 0),
                 "bertScore": (0., 0)}
    # Training
    if args.do_train:
        train_dataset = torch.load(args.data_file)
        train_dataset = train_dataset["dataset"]
        global_step, tr_loss, best_ckpt = train(args, train_dataset, model, tokenizer, best_ckpt)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        logger.info("Saving final model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            if global_step == '1':
                continue
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            results = evaluate(args, model, args.dev_data_file, args.dev_eval_file, args.dev_source_file,
                               tokenizer, result_file=f"{checkpoint}/dev.res",
                               score_file=f"{checkpoint}/dev.score", prefix=prefix)
            for key in results:
                if results[key] > best_ckpt[key][0]:
                    best_ckpt[key] = (results[key], checkpoint)

    # save best checkpoints
    if args.do_train or args.do_eval:
        with open(f"{args.output_dir}/best_ckpt.json", 'w') as f:
            json.dump(best_ckpt, f)

    # testing
    if args.do_test:
        if not args.test_checkpoint:
            logger.info("test checkpoint has not been specified, so load from best_ckpt.json...")
            with open(f"{args.output_dir}/best_ckpt.json", 'r') as f:
                best = json.load(f)
                print("best_ckpt:", best)
        else:
            checkpoint = args.test_checkpoint
            best = {"best": (0., checkpoint)}

        test_res = {}
        for key in best:
            checkpoint = os.path.join(args.output_dir, "checkpoint-{}".format(key))
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            res = evaluate(args, model, args.test_data_file, args.test_eval_file, args.test_source_file,
                           tokenizer, result_file=f"{args.output_dir}/{args.test_res}_{key}.res",
                           score_file=f"{args.output_dir}/{args.test_res}_{key}.score",
                           prefix=prefix, test=True, dev_eval_file1=args.test_eval_file1)
            test_res[key] = res[key]
        print(test_res)
        with open(f"{args.output_dir}/best_ckpt_test_2ref.json" if args.test_eval_file1
                  else f"{args.output_dir}/best_ckpt_test.json", 'w') as f:
            json.dump(test_res, f)


if __name__ == "__main__":
    main()