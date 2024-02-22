from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training , LoraConfig, get_peft_model
import urllib3
import os

urllib3.disable_warnings()

os.environ['CURL_CA_BUNDLE'] = ''

dataset = load_dataset("yatsby/persona_chat")

print(dataset['train'][0]['text'])

model_id = "beomi/polyglot-ko-12.8b-safetensors"
# model_id = "beomi/polyglot-ko-5.8b"

#bitsandbytes quantization => 모델 크기 자체를 줄임
bnb_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_use_double_quant = True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config,device_map={"":0})

dataset = dataset.map(lambda samples:tokenizer(samples['text']), batched = True)

#gradient checkpointing 으로 메모리 줄이기
model.gradient_checkpointing_enable()

#peft 준비 with quantization
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['query_key_value'],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model,config)

#peft 후 trainable parameter 몇개인지 확인
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params+= param.numel()
            
    print(f"trainable_parameters : {trainable_params}, all_parameters : {all_params}")
    
print_trainable_parameters(model)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall' : recall,
        'auroc' : auc
    }



trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    # eval_dataset=dataset['valid'],
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        # per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        # max_steps=50,
        learning_rate=1e-4, # 주로 2e-5 -> 0.00002 
        fp16=True,  # 연산 값들을 fp16으로 바꿈, bf16=True 도 사용 , 사전학습 모델 config dtype 따라가는게 좋았음
        logging_steps=50,
        # eval_steps=10,
        output_dir="./lora_finetuning",
        # optim="paged_adamw_8bit"
        optim="adafactor" # adamw_torch 많이 쓰지만 adafactor 가 메모리 덜 차지함
    ),
    data_collator= DataCollatorForLanguageModeling(tokenizer,mlm=False)
)
model.config.use_cache = False
trainer.train()

model.eval()
model.config.use_cache = True



def gen(p, x):
    gened = model.generate(
        **tokenizer(
            f"###페르소나: {p}\n\n###질문: {x}\n\n###답변:", 
            return_tensors='pt', 
            return_token_type_ids=False
        ), 
        max_new_tokens=512,
        # temperature=0.7,
        # early_stopping=True,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(gened[0]))
    
gen("{'나이' : '22세', '비밀': '선천적 배고픔', '성격': '밝고 유쾌한 성격, 매사에 친절함', '외모': '검정색 긴 머리, '건강한 몸', '이름': '민경', '이상': '많은 돈을 벌어 워라밸을 누리는 것, 사랑하는 사람들과 사이좋게 지내는 것','직업': '직장인'}",'요즘은 무슨 생각하면서 지내??')
