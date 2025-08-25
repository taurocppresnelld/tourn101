#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import pathlib
import torch
import pandas as pd
import time

import toml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def format_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, remaining_seconds)


def calculate_avg_loss_from_file(task_id: str):
    # Read log file
    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
    with open(filelog, 'r') as f:
        log_text = f.read()

    # Extract avr_loss values
    losses = [float(x) for x in re.findall(r'avr_loss=([\d.]+)', log_text)]
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    
    return avg_loss


def calculate_avg_time_from_file(task_id: str):
    import re

    # Read log file
    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
    with open(filelog, 'r') as f:
        log_text = f.read()

    # Extract iteration times (value before 's/it')
    times = [float(x) for x in re.findall(r'([\d.]+)s/it', log_text)]
    
    avg_time = sum(times) / len(times) if times else 0.0
    
    return avg_time


def parse_runtime_logs(task_id: str):
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
    
    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
    with open(filelog, 'r') as f:
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


def parse_loss_logs(task_id: str):
    import re
    import ast

    """
    Parses a log file and extracts JSON-like loss entries.
    Each entry should look like:
    {'loss': 1.2788, 'grad_norm': 0.22516657412052155, 'learning_rate': 9e-06, 'epoch': 0.01}
    
    Returns:
        List of dicts containing the parsed entries.
    """
    pattern = re.compile(r"\{['\"]loss['\"].*?\}")
    entries = []
    
    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
    with open(filelog, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                entry_str = match.group(0)
                try:
                    # Safely evaluate the JSON-like dict string
                    entry = ast.literal_eval(entry_str)
                    # print(f"{entry}")
                    if entry['learning_rate'] > 0.0:
                        entries.append(entry)
                except (ValueError, SyntaxError):
                    # Skip lines that don't parse correctly
                    continue
    return entries


def get_image_training_config_template_path(model_type: str, level="win") -> str:
    model_type = model_type.lower()
    if model_type == ImageModelType.SDXL.value:
        return str(pathlib.Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / f"base_sdxl_{level}.toml")
    elif model_type == ImageModelType.FLUX.value:
        return str(pathlib.Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / f"base_flux_{level}.toml")


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path


def create_config(task_id, model, model_type, expected_repo_name=None, hours_to_complete=2, is_warmup=True, level="win", batch=32, seq=1024, lrate=0.0002, runtime=10, elaptime=0):
    # time_percent = 0.89
    # time_limit = 15
    time_percent = 0.83
    time_limit = 10

    warmup_percent = 0.10
    warmup_limit = 5
    warmup_step = 5

    """Create the diffusion config file"""
    config_template_path = get_image_training_config_template_path(model_type, level)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    config['train_batch_size'] = int(batch/config['gradient_accumulation_steps'])
    config['resolution'] = seq
    config['learning_rate'] = lrate

    # Update config
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = train_paths.get_image_training_images_dir(task_id)
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir


    print(f"current_config: {config}")


    print(f"Total hours {hours_to_complete}")


    trainable_params = 5000000000
    trainable_params_hour = int(trainable_params/hours_to_complete)

    try:
        model_param_path = os.path.join("/workspace/axolotl", "model_param.json")
        with open(model_param_path, 'r') as f:
            data_models = json.load(f)

            for data in data_models:
                if data['model_name'].lower() == model.lower():
                    print(f"model: {data['model_name']}")

                    trainable_params = int(data['trainable_params'])
                    trainable_params_hour = int(trainable_params/hours_to_complete)
                    all_params = int(data['all_params'])
                    trainable_percent = data['trainable_percent']
                    print(f"trainable_params: {trainable_params}")
                    print(f"trainable_params_hour: {trainable_params_hour}")
                    print(f"all_params: {all_params}")
                    print(f"trainable_percent: {trainable_percent}")

        # config = customize_config(config, task_type, model, model_path, all_params)

    except Exception as e:
        print(f"Error checking and logging base model size: {e}")


    if is_warmup:
        config['max_train_steps'] = warmup_step
        config['lr_warmup_steps'] = warmup_step

    else:
        max_steps_percent_limit = int((hours_to_complete*60*60*time_percent-(warmup_limit*60))-elaptime)
        max_steps_percent_percent = int((hours_to_complete*60*60*time_percent-(hours_to_complete*60*60*warmup_percent))-elaptime)
        max_steps_limit_limit = int((hours_to_complete*60*60-(time_limit*60)-(warmup_limit*60))-elaptime)
        max_steps_limit_percent = int((hours_to_complete*60*60-(time_limit*60)-(hours_to_complete*60*60*warmup_percent))-elaptime)

        my_warmup = [max_steps_percent_limit, max_steps_percent_percent, max_steps_limit_limit, max_steps_limit_percent]
        my_warmup_min = max(my_warmup)
        
        config['max_train_steps'] = int(my_warmup_min/runtime)

        print(f"Final time {format_seconds(my_warmup_min)}")

    print(f"max_train_steps: {config['max_train_steps']}")
    

    # config['max_train_steps'] = 0
    # config['max_train_steps'] = 10
    # config['max_train_steps'] = 20

    print(f"max_train_steps: {config['max_train_steps']}")


    # if config['lr_warmup_steps'] > config['max_train_steps']:
    #     config['lr_warmup_steps'] = config['max_train_steps']


    config['tokenizer_name'] = "/cache/models/models--openai--clip-vit-large-patch14"


    print(f"custom_config: {config}")


    # Save config to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(task_id, model, model_type, expected_repo_name, hours_to_complete=2):
    start_time = time.time()

    docker_level = ["win1","win2","win3","win4"]
    # docker_batch = [8,8,8,4,4,4]
    docker_batch = [8,8,8,4,4,4,1,1,1]
    docker_seq = ["1024,1024","768,768","512,512","1024,1024","768,768","512,512","1024,1024","768,768","512,512","1024,1024","768,768","512,512"]
    docker_lrate = 0.0002
    docker_runtime = 10
    docker_config = {}

    docker_failed = True
    idx = 0
    bdx = 0

    # time_percent = 0.89
    # time_limit = 15
    time_percent = 0.83
    time_limit = 10


    try:
        while docker_failed:
            docker_error = ""
            dummy_batch = docker_batch[bdx]
            dummy_batch = dummy_batch - (dummy_batch % 4)
            if dummy_batch < 1:
                dummy_batch = 1

            end_time = time.time()
            elapsed_time = end_time - start_time

            config_path = create_config(
                task_id,
                model,
                model_type,
                expected_repo_name,
                hours_to_complete,
                is_warmup=True,
                level=docker_level[idx],
                batch=dummy_batch,
                seq=docker_seq[bdx],
                lrate=docker_lrate,
                runtime=docker_runtime,
                elaptime=elapsed_time
            )

            try:
                print(f"Docker WARMUP ===============================")


                print(f"Starting training with config: {config_path}", flush=True)
                """Run the training process using the specified config file."""
                with open(config_path, "r") as file:
                    config = toml.load(file)

                print(f"Starting training with level: {docker_level[idx]}", flush=True)
                print(f"Starting training with gradient: {config['gradient_accumulation_steps']}", flush=True)
                print(f"Starting training with batch: {config['train_batch_size']}", flush=True)
                print(f"Starting training with seq: {config['resolution']}", flush=True)
                print(f"Starting training with lrate: {config['learning_rate']}", flush=True)

                # training_command = [
                #     "accelerate", "launch",
                #     "--dynamo_backend", "no",
                #     "--dynamo_mode", "default",
                #     "--mixed_precision", "bf16",
                #     "--num_processes", "1",
                #     "--num_machines", "1",
                #     "--num_cpu_threads_per_process", "2",
                #     f"/app/sd-scripts/{model_type}_train_network.py",
                #     "--config_file", config_path
                # ]

                # training_command = f"huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential; wandb login $WANDB_TOKEN; accelerate launch -m axolotl.cli.train {config_path}" 

                # training_command = f"accelerate launch -m axolotl.cli.train {config_path}" 

                training_command = f"accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision bf16 --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 /app/sd-scripts/{model_type}_train_network.py --config_file {config_path}"

                print("Starting training subprocess...\n", flush=True)
                
                process = subprocess.Popen(
                    training_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )


                filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
                with open(filelog, "w") as f:
                    for line in process.stdout:
                        f.write(line)
                        f.flush()

                        print(line, end="", flush=True)

                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        if "CUDA out of memory" in line:
                            docker_error = "OutOfMemoryError"
                            sys.exit(docker_error) 
                        elif "Caching is incompatible with gradient" in line:
                            docker_error = "Cachingisincompatible"
                            sys.exit(docker_error) 
                        elif "get_max_length" in line:
                            docker_error = "Getmaxlength"
                            sys.exit(docker_error) 
                        elif "mat1 and mat2 must have the same dtype" in line:
                            docker_error = "Musthavethesamedtype"
                            sys.exit(docker_error) 
                        elif "but found Float" in line:
                            docker_error = "ButfoundFloat"
                            sys.exit(docker_error) 
                        elif "tuple index out of range" in line:
                            docker_error = "Tupleindexoutofrange"
                            sys.exit(docker_error) 
                        elif "list index out of range" in line:
                            docker_error = "Listindexoutofrange"
                            sys.exit(docker_error) 
                        elif "DPOTrainer.create_model_card" in line:
                            docker_error = "Dpotrainermodelcard"
                            sys.exit(docker_error) 
                        elif "This might be caused by insufficient shared memory" in line:
                            docker_error = "Insufficientshared"
                            sys.exit(docker_error) 
                        elif "Signals.SIGKILL" in line:
                            docker_error = "Signalskill"
                            sys.exit(docker_error) 
                        elif elapsed_time > int(hours_to_complete*60*60*time_percent):
                            docker_error = "Outoftimepercent"
                            sys.exit(docker_error) 
                        elif elapsed_time > int((hours_to_complete*60*60)-(time_limit*60)):
                            docker_error = "Outoftimelimit"
                            sys.exit(docker_error) 


                return_code = process.wait()
                if return_code != 0:
                    if "OutOfMemoryError" in docker_error:
                        raise torch.OutOfMemoryError()
                    else:
                        raise subprocess.CalledProcessError(return_code, training_command)

                print("Training subprocess completed successfully.", flush=True)


                docker_failed = False


            except SystemExit as e:
                if "OutOfMemoryError" in docker_error:
                    print("Training subprocess OutOfMemoryError!", flush=True)
                    if bdx < len(docker_batch):
                        bdx = bdx + 1
                        if bdx >= len(docker_batch):
                            bdx = len(docker_batch)-1
                    if dummy_batch <= 4:
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                    docker_failed = True
                elif "Insufficientshared" in docker_error:
                    print("Training subprocess Insufficientshared!", flush=True)
                    if bdx < len(docker_batch):
                        bdx = bdx + 1
                        if bdx >= len(docker_batch):
                            bdx = len(docker_batch)-1
                    if dummy_batch <= 4:
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                    docker_failed = True
                elif "Signalskill" in docker_error:
                    print("Training subprocess Signalskill!", flush=True)
                    if bdx < len(docker_batch):
                        bdx = bdx + 1
                        if bdx >= len(docker_batch):
                            bdx = len(docker_batch)-1
                    if dummy_batch <= 4:
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                    docker_failed = True
                elif "Cachingisincompatible" in docker_error:
                    print("Training subprocess Cachingisincompatible!", flush=True)
                    docker_config['gradient_checkpointing']= False
                    docker_failed = True
                elif "Getmaxlength" in docker_error:
                    print("Training subprocess Getmaxlength!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "Musthavethesamedtype" in docker_error:
                    print("Training subprocess Musthavethesamedtype!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "ButfoundFloat" in docker_error:
                    print("Training subprocess ButfoundFloat!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "Tupleindexoutofrange" in docker_error:
                    print("Training subprocess Tupleindexoutofrange!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "Listindexoutofrange" in docker_error:
                    print("Training subprocess Listindexoutofrange!", flush=True)
                    idx = 0
                    bdx = 0
                    docker_failed = True
                elif "Dpotrainermodelcard" in docker_error:
                    print("Training subprocess Dpotrainermodelcard!", flush=True)
                    docker_failed = False
                elif "Outoftimepercent" in docker_error:
                    print("Training subprocess Outoftimepercent!", flush=True)
                    docker_failed = False
                elif "Outoftimelimit" in docker_error:
                    print("Training subprocess Outoftimelimit!", flush=True)
                    docker_failed = False


            except subprocess.CalledProcessError as e:
                print("Training subprocess failed!", flush=True)
                print(f"Exit Code: {e.returncode}", flush=True)
                print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)

                print("Training subprocess unknown!", flush=True)
                idx = idx + 1
                if idx >= len(docker_level):
                    idx = len(docker_level)-1
                docker_failed = True

                # raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


    except Exception as e:
        print(f"Error processing job main: {str(e)}")

    finally:
        print(f"Docker WARMUP finally ===============================")

        try:
            docker_runtime = calculate_avg_time_from_file(task_id)
            print(f"docker_runtime: {docker_runtime}")

            if model_type == ImageModelType.SDXL.value:
                docker_runtime =  int(docker_runtime*1.1)
            elif model_type == ImageModelType.FLUX.value:
                docker_runtime =  int(docker_runtime*0.95)

            print(f"Avg runtime: {docker_runtime}")

        except Exception as e:
            print(f"Failed to get avg runtime: {e}")


        docker_failed = True

        try:
            while docker_failed:
                docker_error = ""
                dummy_batch = docker_batch[bdx]
                dummy_batch = dummy_batch - (dummy_batch % 4)
                if dummy_batch < 1:
                    dummy_batch = 1

                end_time = time.time()
                elapsed_time = end_time - start_time

                config_path = create_config(
                    task_id,
                    model,
                    model_type,
                    expected_repo_name,
                    hours_to_complete,
                    is_warmup=False,
                    level=docker_level[idx],
                    batch=dummy_batch,
                    seq=docker_seq[bdx],
                    lrate=docker_lrate,
                    runtime=docker_runtime,
                    elaptime=elapsed_time
                )

                try:
                    print(f"Docker TRAINING ===============================")


                    print(f"Starting training with config: {config_path}", flush=True)
                    """Run the training process using the specified config file."""
                    with open(config_path, "r") as file:
                        config = toml.load(file)

                    print(f"Starting training with level: {docker_level[idx]}", flush=True)
                    print(f"Starting training with gradient: {config['gradient_accumulation_steps']}", flush=True)
                    print(f"Starting training with batch: {config['train_batch_size']}", flush=True)
                    print(f"Starting training with seq: {config['resolution']}", flush=True)
                    print(f"Starting training with lrate: {config['learning_rate']}", flush=True)

                    # training_command = [
                    #     "accelerate", "launch",
                    #     "--dynamo_backend", "no",
                    #     "--dynamo_mode", "default",
                    #     "--mixed_precision", "bf16",
                    #     "--num_processes", "1",
                    #     "--num_machines", "1",
                    #     "--num_cpu_threads_per_process", "2",
                    #     f"/app/sd-scripts/{model_type}_train_network.py",
                    #     "--config_file", config_path
                    # ]

                    # training_command = f"huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential; wandb login $WANDB_TOKEN; accelerate launch -m axolotl.cli.train {config_path}" 

                    # training_command = f"accelerate launch -m axolotl.cli.train {config_path}" 

                    training_command = f"accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision bf16 --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 /app/sd-scripts/{model_type}_train_network.py --config_file {config_path}"

                    print("Starting training subprocess...\n", flush=True)
                    
                    process = subprocess.Popen(
                        training_command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )


                    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
                    with open(filelog, "w") as f:
                        for line in process.stdout:
                            f.write(line)
                            f.flush()

                            print(line, end="", flush=True)

                            end_time = time.time()
                            elapsed_time = end_time - start_time

                            if "CUDA out of memory" in line:
                                docker_error = "OutOfMemoryError"
                                sys.exit(docker_error) 
                            elif "Caching is incompatible with gradient" in line:
                                docker_error = "Cachingisincompatible"
                                sys.exit(docker_error) 
                            elif "get_max_length" in line:
                                docker_error = "Getmaxlength"
                                sys.exit(docker_error) 
                            elif "mat1 and mat2 must have the same dtype" in line:
                                docker_error = "Musthavethesamedtype"
                                sys.exit(docker_error) 
                            elif "but found Float" in line:
                                docker_error = "ButfoundFloat"
                                sys.exit(docker_error) 
                            elif "tuple index out of range" in line:
                                docker_error = "Tupleindexoutofrange"
                                sys.exit(docker_error) 
                            elif "list index out of range" in line:
                                docker_error = "Listindexoutofrange"
                                sys.exit(docker_error) 
                            elif "DPOTrainer.create_model_card" in line:
                                docker_error = "Dpotrainermodelcard"
                                sys.exit(docker_error) 
                            elif "This might be caused by insufficient shared memory" in line:
                                docker_error = "Insufficientshared"
                                sys.exit(docker_error) 
                            elif "Signals.SIGKILL" in line:
                                docker_error = "Signalskill"
                                sys.exit(docker_error) 
                            elif elapsed_time > int(hours_to_complete*60*60*time_percent):
                                docker_error = "Outoftimepercent"
                                sys.exit(docker_error) 
                            elif elapsed_time > int((hours_to_complete*60*60)-(time_limit*60)):
                                docker_error = "Outoftimelimit"
                                sys.exit(docker_error) 


                    return_code = process.wait()
                    if return_code != 0:
                        if "OutOfMemoryError" in docker_error:
                            raise torch.OutOfMemoryError()
                        else:
                            raise subprocess.CalledProcessError(return_code, training_command)

                    print("Training subprocess completed successfully.", flush=True)


                    docker_failed = False


                except SystemExit as e:
                    if "OutOfMemoryError" in docker_error:
                        print("Training subprocess OutOfMemoryError!", flush=True)
                        if bdx < len(docker_batch):
                            bdx = bdx + 1
                            if bdx >= len(docker_batch):
                                bdx = len(docker_batch)-1
                        if dummy_batch <= 4:
                            idx = idx + 1
                            if idx >= len(docker_level):
                                idx = len(docker_level)-1
                        docker_failed = True
                    elif "Insufficientshared" in docker_error:
                        print("Training subprocess Insufficientshared!", flush=True)
                        if bdx < len(docker_batch):
                            bdx = bdx + 1
                            if bdx >= len(docker_batch):
                                bdx = len(docker_batch)-1
                        if dummy_batch <= 4:
                            idx = idx + 1
                            if idx >= len(docker_level):
                                idx = len(docker_level)-1
                        docker_failed = True
                    elif "Signalskill" in docker_error:
                        print("Training subprocess Signalskill!", flush=True)
                        if bdx < len(docker_batch):
                            bdx = bdx + 1
                            if bdx >= len(docker_batch):
                                bdx = len(docker_batch)-1
                        if dummy_batch <= 4:
                            idx = idx + 1
                            if idx >= len(docker_level):
                                idx = len(docker_level)-1
                        docker_failed = True
                    elif "Cachingisincompatible" in docker_error:
                        print("Training subprocess Cachingisincompatible!", flush=True)
                        docker_config['gradient_checkpointing']= False
                        docker_failed = True
                    elif "Getmaxlength" in docker_error:
                        print("Training subprocess Getmaxlength!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "Musthavethesamedtype" in docker_error:
                        print("Training subprocess Musthavethesamedtype!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "ButfoundFloat" in docker_error:
                        print("Training subprocess ButfoundFloat!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "Tupleindexoutofrange" in docker_error:
                        print("Training subprocess Tupleindexoutofrange!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "Listindexoutofrange" in docker_error:
                        print("Training subprocess Listindexoutofrange!", flush=True)
                        idx = 0
                        bdx = 0
                        docker_failed = True
                    elif "Dpotrainermodelcard" in docker_error:
                        print("Training subprocess Dpotrainermodelcard!", flush=True)
                        docker_failed = False
                    elif "Outoftimepercent" in docker_error:
                        print("Training subprocess Outoftimepercent!", flush=True)
                        docker_failed = False
                    elif "Outoftimelimit" in docker_error:
                        print("Training subprocess Outoftimelimit!", flush=True)
                        docker_failed = False


                except subprocess.CalledProcessError as e:
                    print("Training subprocess failed!", flush=True)
                    print(f"Exit Code: {e.returncode}", flush=True)
                    print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)

                    print("Training subprocess unknown!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True

                    # raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


        except Exception as e:
            print(f"Error processing job main: {str(e)}")

        finally:
            print(f"Docker TRAINING finally ===============================")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    # # Create config file
    # config_path = create_config(
    #     args.task_id,
    #     model_path,
    #     args.model_type,
    #     args.expected_repo_name,
    # )

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # # Run training
    # run_training(args.model_type, config_path)

    run_training(
        args.task_id,
        model_path,
        args.model_type,
        args.expected_repo_name,
        args.hours_to_complete
    )


if __name__ == "__main__":
    asyncio.run(main())
