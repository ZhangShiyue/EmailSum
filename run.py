import os
import argparse
import ujson as json

parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    default=None,
    type=str,
    required=True,
    help="Task name",
)
parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    required=True,
    help="data directory",
)
parser.add_argument(
    "--model_type",
    default="t5",
    type=str,
    help="Model type name",
)
parser.add_argument(
    "--model",
    default="t5-base",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed",
)
parser.add_argument(
    "--min_output_length",
    default=5,
    type=int,
    help="minimum output length during decoding",
)
parser.add_argument(
    "--max_turn_length",
    default=10,
    type=int,
    help="maximum turn length",
)
parser.add_argument(
    "--max_per_turn_length", default=1, type=int,
    help="The maximum length of input per turn.",
)
parser.add_argument(
    "--max_input_length",
    default=512,
    type=int,
    help="maximum input length at each step",
)
parser.add_argument(
    "--max_output_length",  # this is not the same as the max_output_length in t5.py
    default=56,
    type=int,
    help="maximum output length at each step",
)
parser.add_argument(
    "--dec_max_output_length",  # this is the same as the max_output_length in t5.py
    default=200,
    type=int,
    help="maximum output length during decoding",
)
parser.add_argument(
    "--train_batch_size",
    default=4,
    type=int,
    help="Train batch size",
)
parser.add_argument(
    "--eval_batch_size",
    default=4,
    type=int,
    help="Eval batch size",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=32,
    type=int,
    help="Gradient accumulation steps",
)
parser.add_argument(
    "--num_train_epochs",
    default=70,
    type=int,
    help="Number of training epochs",
)
parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--memory_type", default="base", type=str, help="The type of model, base or ht5")
parser.add_argument("--scheduler", default="linear", type=str, help="Scheduler name")
parser.add_argument("--optimizer", default="adam", type=str, help="The optimizer to use")
parser.add_argument("--warmup_steps", default=0., type=float, help="Linear warmup over warmup_steps.")
parser.add_argument("--zero_shot", action="store_true", help="Whether to update parameter.")
parser.add_argument("--from_pretrain", action="store_true", help="Load from pretrained model")
parser.add_argument("--pretrained_checkpoint", default=None, type=str, help="Load from pretrained model")
parser.add_argument("--from_scratch", action="store_true", help="Load from pretrained model")
parser.add_argument("--test_only", action="store_true", help="Whether only test")
args = parser.parse_args()

data_dir = args.data_dir
output_dir = f"train/{args.model_type}_{args.task}_seed{args.seed}_{args.memory_type}_{args.model}"

os.system(f"python3 t5.py --model_type {args.model_type} --model_name_or_path {args.model} "
          f"{f'--from_pretrain --pretrained_checkpoint {args.pretrained_checkpoint}' if args.from_pretrain else ''} "
          f"{'--from_scratch' if args.from_scratch else ''} "
          f"--task_name {args.task} {'' if args.test_only else '--do_train --do_eval'} --do_test "
          f"--data_dir {data_dir} "
          f"--data_file {data_dir}/train_{args.max_input_length}_{args.max_output_length}.t5 "
          f"--eval_file {data_dir}/train.target --source_file {data_dir}/train.source "
          f"--dev_data_file {data_dir}/dev_{args.max_input_length}_{args.max_output_length}.t5 "
          f"--dev_eval_file {data_dir}/dev.target --dev_source_file {data_dir}/dev.source  "
          f"--test_data_file {data_dir}/test_{args.max_input_length}_{args.max_output_length}.t5 "
          f"--test_eval_file {data_dir}/test.target --test_source_file {data_dir}/test.source "
          f"--cache_dir train/cache --max_input_length {args.max_input_length} --do_lower_case "
          f"--min_output_length {args.min_output_length} --max_output_length {args.dec_max_output_length} "
          f"--max_turn_length {args.max_turn_length} "
          f"--per_gpu_eval_batch_size={args.eval_batch_size} --per_gpu_train_batch_size={args.train_batch_size} "
          f"--learning_rate {args.learning_rate} --warmup_steps {args.warmup_steps} "
          f"--scheduler {args.scheduler} --optimizer {args.optimizer} "
          f"--gradient_accumulation_steps {args.gradient_accumulation_steps} --num_train_epochs {args.num_train_epochs} "
          f"--output_dir {output_dir} --memory_type {args.memory_type} "
          f"--overwrite_output_dir --evaluate_during_training --seed {args.seed} ")
