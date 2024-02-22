from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset, Dataset, DatasetDict

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params+= param.numel()
            
    print(f"trainable_parameters : {trainable_params}, all_parameters : {all_params}")
    
from peft import PeftModel, PeftConfig

peft_model_id = "yatsby/qlora-gemini-persona-qna-finetuned"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
print_trainable_parameters(model)
model.eval()

def gen(p, x):
    gened = model.generate(
        **tokenizer(
            f"### 페르소나 : {p}\n\n### 질문: {x}\n\n### 답변:", 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to('cuda'), 
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))
    
gen("{'나이' : '22세', '비밀': '선천적 배고픔', '성격': '밝고 유쾌한 성격, 매사에 친절함', '외모': '검정색 긴 머리, '건강한 몸매', '이름': '민경', '이상': '많은 돈을 벌어 워라밸을 누리는 것, 사랑하는 사람들과 사이좋게 지내는 것','직업': '직장인'}",'요즘은 무슨 생각하면서 지내??')