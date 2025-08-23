import torch

from model_utility import get_model_architecture, get_model_num_params, get_use_liger, disable_flash_attention, get_data_size, get_gpu_count
from copy import deepcopy

distributed = "ddp"
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        distributed = "ds"

FIXED_BS_CONFIG = {
    "EleutherAI/gpt-neo-1.3B": {"batch_size": 36},
    "EleutherAI/gpt-neo-125m": {"batch_size": 48},
    "bigscience/bloom-560m": {"batch_size": 10},
    "facebook/opt-1.3b" : {"batch_size": 38},
    "facebook/opt-350m": {"batch_size": 36},
    "facebook/opt-125m": {"batch_size": 48},
}

INSTRUCT_CONFIG = {
    "0_1_b": {
        "lr": 0.0001,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 140,
        "use_lora": False
    },
    "1_2_b": {
        "lr": 0.0001,
        "distributed": "ddp",
        "gpu_count": 1,
        "use_lora": False,
        "batch_size": 100,
    },
    "2_4_b": {
        "lr": 7.5e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 48,
    },
    "4_5_b": {
        "lr": 7e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 40,
    },
    "5_9_b": {
        "lr": 3.5e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 28,
    },
    "9_12_b": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 2,
        "use_lora": True,
        "batch_size": 32,
    },
    "12_15_b": {
        "lr": 1e-4,
        "distributed": distributed,
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 30,
    },
    "15_40_b": {
        "lr": 8e-5,
        "distributed": distributed,
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 18,
    },
    "40_80_b": {
        "lr": 8e-5,
        "distributed": distributed,
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 8,
    }        
}

for key in INSTRUCT_CONFIG:
    INSTRUCT_CONFIG[key]["label"] = key
    

def get_instruct_config(param_nums: int) -> dict:
    result = {
            "lr": 4e-5,
            "distributed": distributed,
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }
    if param_nums < 1_000_000_000:
        result = INSTRUCT_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = INSTRUCT_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = INSTRUCT_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = INSTRUCT_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = INSTRUCT_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = INSTRUCT_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:  
        result = INSTRUCT_CONFIG["12_15_b"]
    elif param_nums < 35_000_000_000:
        result = INSTRUCT_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = INSTRUCT_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
    result = deepcopy(result)
    if param_nums < 9_000_000_000 and param_nums > 8_000_000_000:
        result["batch_size"] = int(2 * result["batch_size"] / 3)
    return result


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
        "max_steps",
        "warmup_steps",
        "save_steps",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_instruct.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --max_steps {max_steps} \
    --warmup_steps {warmup_steps} \
    --save_steps {save_steps} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} \
    --packing {packing} --disable_fa {disable_fa}"""
    )
    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    if config["use_lora"]:
        template = template + """ --use_lora True"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def get_learning_rate(config: dict, trainable_params: int):
    """
    Auto-calculate learning rate based on trainable parameter count 
    and training type (instruct, dpo, grpo).
    
    Args:
        trainable_params (int): Number of trainable parameters (e.g., LoRA params).
        training_type (str): One of ['instruct', 'dpo', 'grpo'].
    
    Returns:
        float: Suggested learning rate.
    """

    # Base coefficient for scaling (adjust per training type)

    c = 5e-2   # more forgiving

    # if training_type.lower() == "instruct":
    #     c = 5e-2   # more forgiving
    # elif training_type.lower() == "dpo":
    #     c = 3e-2   # slightly lower than instruct
    # elif training_type.lower() == "grpo":
    #     c = 1.5e-2 # RL-based, more sensitive
    # else:
    #     raise ValueError("training_type must be 'instruct', 'dpo', or 'grpo'.")

    c = config['lr']
    print(f"base lr {c}")

    # Scale LR inversely with sqrt of params
    lr = c / (trainable_params ** 0.5)
    print(f"scale lr {lr}")

    # Clamp to reasonable bounds for stability

    lr = max(min(lr, 1e-3), 5e-5)

    # if training_type.lower() == "instruct":
    #     lr = max(min(lr, 1e-3), 5e-5)
    # elif training_type.lower() == "dpo":
    #     lr = max(min(lr, 5e-4), 3e-5)
    # elif training_type.lower() == "grpo":
    #     lr = max(min(lr, 2e-4), 1e-5)

    config['lr'] = lr
    print(f"custom lr {config['lr']}")

    return config


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)

    param_nums = train_info["all_params"]

    if not param_nums:
        param_nums = get_model_num_params(model_name, model_path)

    param_nums_trainable = int(param_nums*0.003)
    print(f"param_nums: {param_nums}")
    print(f"param_nums_trainable: {param_nums_trainable}")

    config = get_instruct_config(param_nums)
    print(f"get_instruct_config: {config}")

    # if param_nums_trainable:
    #     config = get_learning_rate(config, param_nums_trainable)
    #     print(f"get_learning_rate: {config}")

    run_config = {
        "epoch_num": 3,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": True,
        "packing": False,
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": 4,
        "max_steps": -1,
        "warmup_steps": 35,
        "save_steps": 200,
    }
    # data_size = get_data_size(train_info["request_path"])
    
    # if run_config["disable_fa"]: # if FA is not usable
    #     run_config["batch_size"] = run_config["batch_size"] * 2

    if model_name in FIXED_BS_CONFIG:
        run_config["batch_size"] = FIXED_BS_CONFIG[model_name]["batch_size"]
    
    if model_architecture.strip().lower() in [
        "gptneoxforcausallm",
        "gptjforcausallm",
        "phiforcausallm",
        "falconforcausallm",
    ]:
        run_config["batch_size"] = int(run_config["batch_size"] // 2)
        if model_name == "EleutherAI/pythia-160m":  # reduce more
            run_config["batch_size"] = int(run_config["batch_size"] / 1.5)
        elif "pythia" in model_name.lower():
            run_config["batch_size"] = int(run_config["batch_size"] / 1.8)
    
    if (
        model_name in ["microsoft/phi-2", "microsoft/phi-1_5"]
    ): 
        run_config["batch_size"] = int(run_config["batch_size"] / 4)
    
    if "bloom-560m" in model_name or "bloomz-560m" in model_name:
        run_config["batch_size"] = 8
    
    if model_name == "mistralai/Mistral-7B-v0.1":
        run_config["batch_size"] = int(3 * run_config["batch_size"] / 4)
    
    if "falcon" in model_name.lower():
        run_config["batch_size"] = int(run_config["batch_size"] / 2)
    
    data_per_step = run_config["batch_size"] * run_config["gpu_nums"]
    if data_per_step >= 64:
        run_config["gradient_accumulation_steps"] = 1
    else:
        run_config["gradient_accumulation_steps"] = int(64 / data_per_step)
    
    print(f"run_config: {run_config}")

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 10
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    
    return {
        "train_request": train_request,
        "run_cmd": run_cmd
    }
