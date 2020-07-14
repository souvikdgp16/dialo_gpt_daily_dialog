import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch

import numpy as np

from os.path import join
from torch.distributed import get_rank, get_world_size

from lsp_model import GPT2LMHeadModel, GPT2Config, Adam
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from gpt2_training.eval_utils import get_model_metrics
from transformers import GPT2Tokenizer

from data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader


from gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

# parser.add_argument("--skip_eval", action='store_true',
#                     help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

# parser.add_argument("--train_batch_size", type=int, default=4,
#                     help="batch size now means per GPU per step")
# parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
#                     help="to increase effective batch size "
#                          "and reduce synchronization")
# parser.add_argument("--eval_batch_size", type=int, default=4)
# parser.add_argument("--learning_rate", type=float, default=1e-5)
# parser.add_argument("--num_optim_steps", type=int, default=1000000,
#                     help="new API specifies num update steps")
# parser.add_argument("--valid_step", type=int, default=10000,
#                     help="how many optim steps between validations")
# parser.add_argument("--warmup_proportion", type=float, default=0.1)
# parser.add_argument("--warmup_steps", type=int, default=16000)

# parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=False)
# parser.add_argument("--lr_schedule", type=str,
#                     choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
# parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True)

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')

parser.add_argument('--emotion', type=boolean_string, default=False)
parser.add_argument('--da', type=boolean_string, default=False)

parser.add_argument('--config', help='JSON config file')


# do normal parsing
args = parser.parse_args()



if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            # PHILLY ENV special cases
            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY',
                              os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY',
                              os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)

if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))


#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2Config.from_json_file(
    join(args.model_name_or_path, 'config.json'))

# if args.local_rank == -1:
#     train_dataloader = BucketingDataLoader(args.train_input_file,
#                                            args.train_batch_size,
#                                            args.max_seq_length)
# else:
#     train_dataloader = DistributedBucketingDataLoader(
#         get_rank(), get_world_size(),
#         args.train_input_file, args.train_batch_size,
#         args.max_seq_length)

# eval_dataloader_loss = DynamicBatchingLoader(
#     args.eval_input_file, enc, args.normalize_data,
#     args.eval_batch_size, args.max_seq_length)

eval_dataloader_loss = BucketingDataLoader(args.eval_input_file, 
        args.eval_batch_size, args.max_seq_length)

# eval_dataloader_gen = get_eval_list_same_length(
#     args.eval_input_file, enc, args.eval_batch_size, True)


#########################################################################
# Prepare Model and Prediction
##########################################################################
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,
                   args, verbose=True)

eval_loss, eval_ppl = eval_model_loss(
                        model, enc, eval_dataloader_loss, args)

