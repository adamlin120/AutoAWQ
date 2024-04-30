import os
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def load_model(model_path):
    model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True, **{"low_cpu_mem_usage": True, "use_cache": False})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def load_sft_data():
    data = load_dataset('yentinglin/TaiwanChat', split="train")
    return data

def concatenate_data(x, tokenizer):
    chat = x['messages']
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": text}

def preprocess_data(data, tokenizer):
    return data.map(lambda x: concatenate_data(x, tokenizer))['text']

def quantize_model(model, tokenizer, quant_config, calib_data):
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    return model

def save_quantized_model(model, tokenizer, quant_path):
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

def main():
    model_path = 'yentinglin/Llama-3-Taiwan-70B-Instruct'
    quant_path = 'Llama-3-Taiwan-70B-Instruct-awq'
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    # Load model
    model, tokenizer = load_model(model_path)

    # Load and preprocess data
    data = load_sft_data()
    calib_data = preprocess_data(data, tokenizer)

    # Quantize model
    quantized_model = quantize_model(model, tokenizer, quant_config, calib_data)

    # Save quantized model
    save_quantized_model(quantized_model, tokenizer, quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')

if __name__ == "__main__":
    main()
