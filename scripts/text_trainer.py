#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import re
import time
from datetime import datetime, timezone, timedelta

import yaml
from transformers import AutoTokenizer


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
# from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
import pathlib



def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None):
    print(f"Running command: {cmd}")
    with open(log_file_path, "w") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)
        
        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)([^\s]+))(\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        return None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)(?P<value>[^\s]+))(\s+)", cmd)
    if match:
        return match.group("value")
    else:
        return None


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"
VAR_ONCE_ERROR = "RuntimeError: Expected to mark a variable ready only once"
CACHING_ERROR = "Caching is incompatible with gradient"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    elif VAR_ONCE_ERROR in text:
        return VAR_ONCE_ERROR
    elif CACHING_ERROR in text:
        return CACHING_ERROR
    else:
        return None


def patch_wandb_symlinks(base_dir:str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass


def parse_runtime_logs(log_path: str):
    import re
    import ast

    """
    Parses a log file and extracts JSON-like loss entries.
    Each entry should look like:
    {'loss': 1.2788, 'grad_norm': 0.22516657412052155, 'learning_rate': 9e-06, 'epoch': 0.01}
    
    Returns:
        List of dicts containing the parsed entries.
    """
    pattern = re.compile(r"\{['\"]train_runtime['\"].*?\}")
    entries = []
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                entry_str = match.group(0)
                try:
                    # Safely evaluate the JSON-like dict string
                    entry = ast.literal_eval(entry_str)
                    # print(f"{entry}")
                    entries.append(entry)
                except (ValueError, SyntaxError):
                    # Skip lines that don't parse correctly
                    continue
    return entries


def format_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, remaining_seconds)


def main():
    start_time = time.time()

    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument(
        "--retries", type=int, help="Number of retries", default=5
    )

    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)


    trainable_params = 5000000000
    all_params = 0

    try:
        model_param_path = os.path.join("./", "model_param.json")
        with open(model_param_path, 'r') as f:
            data_models = json.load(f)

            for data in data_models:
                if data['model_name'].lower() == original_model_name.lower():
                    print(f"model: {data['model_name']}")

                    trainable_params = int(data['trainable_params'])
                    all_params = int(data['all_params'])
                    trainable_percent = data['trainable_percent']
                    print(f"trainable_params: {trainable_params}")
                    print(f"all_params: {all_params}")
                    print(f"trainable_percent: {trainable_percent}")

    except Exception as e:
        print(f"Error checking and logging base model size: {e}")


    # time_percent = 0.89
    # time_limit = 15
    # time_percent = 0.87
    # time_limit = 25
    time_percent = 0.88
    time_limit = 20

    warmup_percent = 0.5
    warmup_limit = 5
    warmup_step = 5

    config_level = ["default","cache","flash","tuple","float","low"]

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))
    log_path = os.path.join(ds_folder, f"train_{args.task_id}.log")


    train_success = False
    cdx = 0

    while not train_success:
        print(
            f"TRAINING {config_level[cdx]} =======================================================================",
            flush=True,
        )


        train_info = {
            "model_name": original_model_name,
            "model_path": model_path,
            "task_id": args.task_id,
            "dataset": dataset_path,
            "hours_to_complete": args.hours_to_complete,
            "expected_repo_name": args.expected_repo_name,
            "end_time": end_time,
            "dataset_type": dataset_type_dict,
            "submission_dir": submission_dir,
            "output_dir": output_dir,
            "min_steps": 100,
            "adjust_batch_size": True,
            "request_path": request_path,
            "max_data_size": args.max_data_size,
            "max_steps": args.max_steps,
            # "max_steps": 20,
            "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
            "all_params": all_params,
        }

        if config_level[cdx] == "default":
            from instruct_config_default import get_training_json as get_instruct_training_json
        elif config_level[cdx] == "cache":
            from instruct_config_cache import get_training_json as get_instruct_training_json
        elif config_level[cdx] == "flash":
            from instruct_config_flash import get_training_json as get_instruct_training_json
        elif config_level[cdx] == "tuple":
            from instruct_config_tuple import get_training_json as get_instruct_training_json
        elif config_level[cdx] == "float":
            from instruct_config_float import get_training_json as get_instruct_training_json
        elif config_level[cdx] == "low":
            from instruct_config_low import get_training_json as get_instruct_training_json
        else:
            from instruct_config import get_training_json as get_instruct_training_json

        if args.task_type == TaskType.INSTRUCTTEXTTASK.value:
            train_info = get_instruct_training_json(train_info)
            tokenize_cmd = (
                f"/workspace/axo_py/bin/python tokenize_instruct.py {request_path}"
            )
            train_cmd = train_info["run_cmd"]

        elif args.task_type == TaskType.DPOTASK.value:
            train_info = get_dpo_training_json(train_info)
            tokenize_cmd = f"python tokenize_dpo.py {request_path}"
            train_cmd = train_info["run_cmd"]

        elif args.task_type == TaskType.GRPOTASK.value:
            train_info = get_grpo_training_json(train_info)
            tokenize_cmd = f"python tokenize_grpo.py {request_path}"
            train_cmd = train_info["run_cmd"]
        else:
            raise ValueError(f"Task type {args.task_type} not supported")

        with open(request_path, "w") as f:
            json.dump(train_info, f, indent=4, ensure_ascii=False)

        run_cmd_with_log(
            tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
        )


        # # sub_train_success = False
        # # i = 0

        # # while not sub_train_success:
        # #     i = i+1

        # #     print(
        # #         f"WARMUP =======================================================================",
        # #         flush=True,
        # #     )
        # #     print(
        # #         f"************* Warmup attempt {i} for task {args.task_id} *************",
        # #         flush=True,
        # #     )
        # #     if i > 0:  # there was something wrong so we will reduce the batch_size
        # #         # first check if the training is OOM
        # #         if os.path.exists(log_path):
        # #             error_type = get_error_type(log_path)
        # #             if error_type == OOM_ERROR:
        # #                 current_batch_size = extract_value_from_cmd(
        # #                     train_cmd, "per_device_train_batch_size"
        # #                 )
        # #                 current_batch_size = int(current_batch_size)
        # #                 if current_batch_size > 1:
        # #                     new_batch_size = current_batch_size // 2
        # #                     print(
        # #                         f"Reducing batch size from {current_batch_size} to {new_batch_size}",
        # #                         flush=True,
        # #                     )
        # #                     train_cmd = replace_args_in_cmd(
        # #                         train_cmd,
        # #                         "per_device_train_batch_size",
        # #                         str(new_batch_size),
        # #                     )
        # #                     print(f"New train command: {train_cmd}", flush=True)
        # #                 else:
        # #                     print(f"batch size is 1, cannot reduce further", flush=True)
        # #                     if args.task_type == TaskType.GRPOTASK.value:
        # #                         # disable vllm
        # #                         train_cmd = replace_args_in_cmd(
        # #                             train_cmd, "use_vllm", "False"
        # #                         )
        # #                         print(f"disable VLLM {train_cmd}", flush=True)
        # #             elif error_type == VLLM_OOM_ERROR:
        # #                 if args.task_type == TaskType.GRPOTASK.value:
        # #                     print(f"VLLM OOM error, disable VLLM", flush=True)
        # #                     train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")

        # #     # empty the log file if it exists
        # #     if os.path.exists(log_path):
        # #         with open(log_path, "w") as f:
        # #             f.write("STARTING WARMUP")

        # #     task_id = args.task_id
        # #     expected_repo_name = args.expected_repo_name
            
        # #     training_env_vars = {
        # #         "WANDB_MODE": "offline",
        # #         "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
        # #         "WANDB_NAME": f"{task_id}_{expected_repo_name}",
        # #     }
            
        # #     print(f"train_cmd: {train_cmd}")

        # #     run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)

        # #     # check if the training is successfully done; it is done, the output_dir should not be empty there is at least 2 files in the submission_dir
        # #     if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        # #         print(f"Warmup failed for task {args.task_id}", flush=True)
        # #     else:
        # #         print(f"Warmup successfully done for task {args.task_id}", flush=True)
        # #         sub_train_success = True
        # #         # break

        # #     # print(f"Warmup successfully done for task {args.task_id}", flush=True)
        # #     # sub_train_success = True
        # #     # # break


        # # end_time = time.time()
        # # elapsed_time = end_time - start_time

        # # try:
        # #     runtimes = parse_runtime_logs(log_path)
        # #     print(f"RUNTIMES: {runtimes}")
        # #     step_runtime = runtimes[0]['train_runtime']/warmup_step

        # #     print(f"AVG RUNTIME: {step_runtime}")

        # #     max_steps_percent_limit = int((args.hours_to_complete*60*60*time_percent-(warmup_limit*60))-elapsed_time)
        # #     max_steps_percent_percent = int((args.hours_to_complete*60*60*time_percent-(args.hours_to_complete*60*60*warmup_percent))-elapsed_time)
        # #     max_steps_limit_limit = int((args.hours_to_complete*60*60-(time_limit*60)-(warmup_limit*60))-elapsed_time)
        # #     max_steps_limit_percent = int((args.hours_to_complete*60*60-(time_limit*60)-(args.hours_to_complete*60*60*warmup_percent))-elapsed_time)

        # #     my_warmup = [max_steps_percent_limit, max_steps_percent_percent, max_steps_limit_limit, max_steps_limit_percent]
        # #     my_warmup_min = max(my_warmup)
        # #     train_steps = int(my_warmup_min/step_runtime)
        # #     print(f"TRAIN STEPS: {train_steps}")
        # #     train_cmd = replace_args_in_cmd(train_cmd, "max_steps", str(train_steps))
        # #     train_info["max_steps"] = train_steps

        # #     train_warmup = int(train_steps*0.20)
        # #     train_cmd = replace_args_in_cmd(train_cmd, "warmup_steps", str(train_warmup))
        # #     train_info["warmup_steps"] = train_warmup
            
        # #     train_save = int(train_steps*0.20)
        # #     train_cmd = replace_args_in_cmd(train_cmd, "save_steps", str(train_save))
        # #     train_info["save_steps"] = train_save

        # #     print(f"FINAL TIME {format_seconds(my_warmup_min)}")

        # # except Exception as e:
        # #     print(f"Failed to get avg runtime: {e}")


        sub_train_success = False
        idx = 0

        while not sub_train_success:
            idx = idx+1

            print(
                f"TRAINING =======================================================================",
                flush=True,
            )
            print(
                f"************* Training attempt {idx} ({config_level[cdx]}) for task {args.task_id} *************",
                flush=True,
            )
            if idx > 0:  # there was something wrong so we will reduce the batch_size
                # first check if the training is OOM
                if os.path.exists(log_path):
                    error_type = get_error_type(log_path)
                    if error_type == OOM_ERROR:
                        current_batch_size = extract_value_from_cmd(
                            train_cmd, "per_device_train_batch_size"
                        )
                        current_batch_size = int(current_batch_size)
                        if current_batch_size > 1:
                            new_batch_size = current_batch_size // 2
                            print(
                                f"Reducing batch size from {current_batch_size} to {new_batch_size}",
                                flush=True,
                            )
                            train_cmd = replace_args_in_cmd(
                                train_cmd,
                                "per_device_train_batch_size",
                                str(new_batch_size),
                            )
                            print(f"New train command: {train_cmd}", flush=True)
                        else:
                            print(f"batch size is 1, cannot reduce further", flush=True)
                            if args.task_type == TaskType.GRPOTASK.value:
                                # disable vllm
                                train_cmd = replace_args_in_cmd(
                                    train_cmd, "use_vllm", "False"
                                )
                                print(f"disable VLLM {train_cmd}", flush=True)
                    elif error_type == VLLM_OOM_ERROR:
                        if args.task_type == TaskType.GRPOTASK.value:
                            print(f"VLLM OOM error, disable VLLM", flush=True)
                            train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")
                    elif error_type == VAR_ONCE_ERROR:
                        print(f"VAR_ONCE_ERROR", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "gradient_checkpointing", "False")
                    elif error_type == CACHING_ERROR:
                        print(f"CACHING_ERROR", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "gradient_checkpointing", "False")

            # empty the log file if it exists
            if os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write("STARTING TRAINING")

            task_id = args.task_id
            expected_repo_name = args.expected_repo_name
            
            training_env_vars = {
                "WANDB_MODE": "offline",
                "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
                "WANDB_NAME": f"{task_id}_{expected_repo_name}",
            }
            
            print(f"train_cmd: {train_cmd}")

            run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)

            # check if the training is successfully done; it is done, the output_dir should not be empty there is at least 2 files in the submission_dir
            if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
                print(f"Training failed for task {args.task_id}", flush=True)
                current_batch_size = extract_value_from_cmd(
                    train_cmd, "per_device_train_batch_size"
                )
                current_batch_size = int(current_batch_size)
                if current_batch_size <= 1:
                    sub_train_success = True
                    train_success = False
                    cdx = cdx+1

            else:
                print(f"Training successfully done for task {args.task_id}", flush=True)
                sub_train_success = True
                train_success = True
                # break

            # print(f"Training successfully done for task {args.task_id}", flush=True)
            # sub_train_success = True
            # # break


    # if not train_success:
    #     print(f"Training failed for task {args.task_id}", flush=True)
    #     # add noise to the model
    #     add_noise_cmd = f"python add_random_noise.py {model_path} {submission_dir}"
    #     run_cmd_with_log(
    #         add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
    #     )

    # # patch_model_metadata(submission_dir, args.model)

    # patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
