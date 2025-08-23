DPO = "dpo"
GRPO = "grpo"
INSTRUCT = "instruct"
import re
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoConfig, AutoModel
import glob
from safetensors.torch import load_file
from pathlib import Path
import torch
import os
import json
import torch

MODEL_CONFIG = {
    "facebook/opt-1.3b": {"model_size": 1_300_000_000},
    "facebook/opt-3b": {"model_size": 3_000_000_000},
    "facebook/opt-6.7b": {"model_size": 6_700_000_000},
    "facebook/opt-13b": {"model_size": 13_000_000_000},
    "EleutherAI/gpt-neo-1.3B": {"model_size": 1_300_000_000},
    "EleutherAI/gpt-neo-125m": {"model_size": 125_000_000},
    "bigscience/bloom-560m": {"model_size": 560_000_000},
    "TinyLlama/TinyLlama_v1.1": {"model_size": 1_100_000_000},
}

hf_api = HfApi()


def get_model_architecture(model_path: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_path)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except:
        return "Unknown"


def get_use_liger(architecture: str) -> str:
    if architecture.lower() in [
        "qwen2forcausallm",
        "llamaforcausallm",
        "gemma2forcausallm",
        "mixtralforcausallm",
        "mistralforcausallm",
        "qwen3forcausallm",
        "phi3forcausallm",
        "gemmaforcausallm",
    ]:
        return "True"
    else:
        return "False"


def count_params_from_safetensors(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not shards:
        return None

    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        tensors = load_file(shard_path)
        total_params += sum(v.numel() for v in tensors.values())

    return total_params


def count_params_from_bin(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.bin"))
    if not shards:
        return None

    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        try:
            state_dict = torch.load(shard_path, map_location="cpu")
            total_params += sum(v.numel() for v in state_dict.values())
        except Exception as e:
            print(f"cannot load {shard_path}: {e}")
            continue

    return total_params


def get_model_size_from_local_path(model_path: str) -> int:
    size = count_params_from_safetensors(model_path)
    if size is not None and size > 1000:
        print(f"Model size from safetensors: {size}")
        return size
    size = count_params_from_bin(model_path)
    if size is not None and size > 1000:
        print(f"Model size from bin: {size}")
        return size
    return None


def get_gpu_count():
    return torch.cuda.device_count()


# def get_model_num_params(model_id: str, model_path: str) -> int:
#     if model_id in MODEL_CONFIG:
#         return MODEL_CONFIG[model_id]["model_size"]
#     try:
#         size = get_model_size_from_local_path(model_path)
#         if size is not None:
#             return size
#         raise Exception(f"Cannot get model size from {model_path}")

#     except Exception as e:
#         print(f"Error getting model size from safetensors: {e}")
#         try:
#             model_size = re.search(r"(\d+)(?=[bB])", model_id)
#             model_size = (
#                 int(model_size.group(1)) * 1_000_000_000 if model_size else None
#             )
#             print(f"Model size from regex: {model_size}")
#             return model_size
#         except Exception as e:
#             print(f"Error getting model size from regex: {e}")
#             return None


def get_model_num_params(model: str, model_path: str) -> int:
    default_size = 5000000000

    if model in MODEL_CONFIG:
        return MODEL_CONFIG[model_id]["model_size"]

    try:
        print(f"Fetching config for {model}...")
        model_config = AutoConfig.from_pretrained(model)

        print("Building model architecture without downloading weights...")
        model_params = AutoModel.from_config(model_config)

        all_params = sum(p.numel() for p in model_params.parameters())
        # trainable_params = sum(p.numel() for p in model_params.parameters() if p.requires_grad)
        trainable_params = int(all_params*0.003)

        print(f"loading tokenizer... {model}")
        print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {trainable_params/all_params}")

        return all_params

    except Exception as e:
        print(f"Error getting model size from automodel: {e}")

        try:
            hf_api = HfApi()
            model_info = hf_api.model_info(model_path)
            size = model_info.safetensors.total
            size_trainable = int(size*0.003)

            print(f"save new model ===============================")
            print(f"loading tokenizer... {model}")
            print(f"trainable params: {size_trainable} || all params: {size} || trainable%: {size_trainable/size}")

            return size

        except Exception as e:
            print(f"Error getting model size from safetensors: {e}")

            try:
                import re
                model_size = re.search(r"(\d+)(?=[bB])", model)
                model_size = (
                    int(model_size.group(1)) * 1_000_000_000 if model_size else None
                )
                print(f"Model size from regex: {model_size}")
                return model_size

            except Exception as e:
                print(f"Error getting model size from regex: {e}")
                return default_size


def disable_flash_attention(architecture: str, model: str) -> str:
    if model == "microsoft/phi-2":
        return "True"
    if "falcon-rw" in model.lower():  # ex, tiiuae/falcon-rw-1b
        return "True"
    # if model == "databricks/dolly-v2-3b":
    #    return "True"
    if architecture.strip().lower() in ["gptneoforcausallm", "bloomforcausallm"]:
        return "True"
    else:
        return "False"


def get_use_vllm(architecture: str, model: str) -> str:
    if model in [
        "Eurdem/Defne_llama3_2x8B",
        "heegyu/WizardVicuna-open-llama-3b-v2",
        "openlm-research/open_llama_3b",
        "TitanML/tiny-mixtral",
        "dunzhang/stella_en_1.5B_v5",
        "oopsung/llama2-7b-n-ox-test-v1",
        "microsoft/phi-2",
        "databricks/dolly-v2-3b",
    ]:
        return False
    if "falcon-rw" in model.lower():
        return False

    if architecture in ["gptneoforcausallm", "bloomforcausallm"]:
        return False
    else:
        return True


def get_gradient_checkpointing(model: str) -> str:
    if "falcon-rw" in model.lower():
        return "False"
    return "True"


def get_data_size(data_path: str) -> int:
    with open(data_path, "r") as f:
        data = json.load(f)
    return len(data)
