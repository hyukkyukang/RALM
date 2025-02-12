import transformers

# Use GPT-small
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token