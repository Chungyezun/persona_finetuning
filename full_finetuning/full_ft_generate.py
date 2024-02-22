import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
# MODEL_NAME = "result_model"
MODEL_NAME = "../../kogpt2_finetuning/checkpoint-2000/"
# MODEL_NAME = "skt/kogpt2-base-v2"
tokenizer_name = "skt/kogpt2-base-v2"
 
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto', low_cpu_mem_usage=True, device_map="auto").eval()
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
 
def gen(prompt):    
    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to("cuda")
    output = model.generate(
        **inputs,
        temperature=0.7,
        do_sample=True,
        max_length=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
 
    return tokenizer.decode(output[0], skip_special_tokens=True)
 
print(gen("###페르소나 : {'나이' : '22세', '비밀': '선천적 배고픔', '성격': '밝고 유쾌한 성격, 매사에 친절함', '외모': '검정색 긴 머리, '건강한 마음', '이름': '민경', '이상': '많은 돈을 벌어 워라밸을 누리는 것, 사랑하는 사람들과 사이좋게 지내는 것','직업': '직장인'}\n\n###질문 : 요즘은 무슨 생각하면서 지내??\n\n###답변:"))

# text = '근육이 커지기 위해서는'
# input_ids = tokenizer.encode(text)
# gen_ids = model.generate(torch.tensor([input_ids]).to("cuda"),
#                            max_length=128,
#                            repetition_penalty=2.0,
#                            pad_token_id=tokenizer.pad_token_id,
#                            eos_token_id=tokenizer.eos_token_id,
#                            bos_token_id=tokenizer.bos_token_id,
#                            use_cache=True)
# generated = tokenizer.decode(gen_ids[0,:].tolist())
# print(generated)