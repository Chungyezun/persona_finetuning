from datasets import load_dataset
import json
from collections import OrderedDict

dataset = load_dataset("yatsby/persona_chat")
file_data = OrderedDict()

for element in dataset['train']:
    file_data["persona"] = element['persona']
    file_data["question"] = element['question']
    file_data["answer"] = element['answer']
    
    with open("persona_data.json", "r", encoding="utf8") as f:
        data = json.load(f)
    data.append(file_data)
    with open("persona_data.json", "w", encoding="utf8") as f:
        json.dump(data,f,ensure_ascii=False,indent="\t")