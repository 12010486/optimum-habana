# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

###############################################################################
# Copyright (C) 2020-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import json
import multiprocessing as mp
import os
import time

import psutil
import torch
import torch.nn.functional as F
from lm_eval import evaluator, utils, tasks
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import get_dtype

# Local imports
from run_generation import setup_parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from utils import finalize_quantization, initialize_model

from optimum.habana.utils import get_hpu_memory_stats


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

eval_logger = utils.eval_logger

# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    pool_size = physical_cpu_count
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if world_size == 0:
        world_size = 1
    pool_size //= world_size
    if (pool_size * world_size) != physical_cpu_count:
        pool_size -= 1
    return spawn_context.Pool(pool_size)


mp.Pool = LimitedSpawnPool


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 384],
    )
    parser.add_argument(
        "--output_path", "-o", type=str, help="Output path with end results and runtime parameters", required=True
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        help="Comma-separated list of task names or task groupings to evaluate on",
        default="hellaswag,lambada_openai,piqa,winogrande",
    )
    parser.add_argument("--limit",
        "-L",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    
    args = setup_parser(parser)

    return args


class HabanaLM(HFLM):

    """
    using the HuggingFace transformers + optimum-habana backend, can run on Intel Gaudi
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        args: argparse.Namespace,
        options: GenerationConfig,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=args.model_name_or_path, device=args.device, **kwargs,)
        self.tokenizer = tokenizer
        self._model = model
        self._batch_size = args.batch_size
        self.buckets: list[int] = sorted(args.buckets)
        self.options = options
        self.device_ = args.device
        self.model_inputs = {"use_cache": self.options.use_cache}
        if self._model.config.model_type in [
            "llama",
            "mistral",
            "falcon",
            "phi",
            "mixtral",
            "qwen2",
            "gptj",
            "starcoder2",
            "gemma",
            "baichuan",
        ]:
            self.model_inputs.update(
                {
                    "reuse_cache": self.options.reuse_cache,
                }
            )
        if self._model.config.model_type in ["llama", "mistral", "qwen2", "falcon", "starcoder2", "gemma", "baichuan"]:
            if self._model.config.model_type != "falcon":
                self.model_inputs.update(
                    {
                        "attn_softmax_bf16": self.options.attn_softmax_bf16,
                    }
                )
            self.model_inputs.update(
                {
                    "use_flash_attention": self.options.use_flash_attention,
                    "flash_attention_recompute": self.options.flash_attention_recompute,
                    "flash_attention_causal_mask": self.options.flash_attention_causal_mask,
                }
            )
        if args.warmup:
            self.warm_up()

    def warm_up(self) -> None:
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self._batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)

    @property
    def eot_token_id(self) -> int:
        return self._model.config.eos_token_id

    @property
    def max_length(self) -> int:
        return self.buckets[-1]

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    def find_bucket(self, length: int) -> list[int]:
        return [b for b in self.buckets if b >= length][0]

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        bs, seq_length = inps.shape
        padding_length = 0
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.options.use_cache and self.options.reuse_cache:
                self._model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self._model.config.pad_token_id)
        logits = self._model(inps.to(self.device_), **self.model_inputs)["logits"].cpu()

        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)
        return logits

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize=False,
        gpus=None,
        max_memory_per_gpu=None,
        max_cpu_memory=None,
        offload_folder="./offload",
        # PEFT, delta weights and quantization options
        peft=None,
        delta=None,
        autogptq=False,
        gptqmodel=False,
        **kwargs,
    ) -> None:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        adapt_transformers_to_gaudi()
        model_kwargs = kwargs if kwargs else {}
        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )


def main() -> None:
    # Modified based on cli_evaluate function in https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/__main__.py/#L268
    args = setup_lm_eval_parser()
    model, _, tokenizer, generation_config = initialize_model(args, eval_logger)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        ) 

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    task_manager = tasks.TaskManager("INFO")

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        print(task_manager.list_all_tasks())
        sys.exit()
    elif args.tasks == "list_groups":
        print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if args.trust_remote_code:
        # trust_remote_code fix was introduced in lm_eval 0.4.3
        eval_logger.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    eval_logger.info(f"Selected Tasks: {task_names}")

    with torch.no_grad():
        lm = HabanaLM(tokenizer, model, args, generation_config)

    eval_start = time.perf_counter()
    with torch.no_grad():
        #results = evaluator.evaluate(lm, lm_tasks, limit=args.limit_iters)
        results = evaluator.simple_evaluate(lm, tasks=task_names,
        num_fewshot=args.num_fewshot,
        device=args.device,
        limit=args.limit,
        write_out=args.write_out,
        log_samples=args.log_samples,
        system_instruction=args.system_instruction,
        task_manager=task_manager,
        predict_only=args.predict_only,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        )

    if args.device == "hpu":
        import habana_frameworks.torch.hpu as torch_hpu

        torch_hpu.synchronize()
    eval_end = time.perf_counter()

    results["args"] = vars(args)
    results["duration"] = eval_end - eval_start

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))

        json.dump(
            results, open(args.output_file, "w"), indent=2, default=utils.handle_non_serializable, ensure_ascii=False
        )

        if args.show_config:
            print(json.dumps(results, indent=2, default=utils.handle_non_serializable, ensure_ascii=False))
    
    if args.quant_config:
        finalize_quantization(model)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization__path)

if __name__ == "__main__":
    main()
