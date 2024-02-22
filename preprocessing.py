	
# -*- coding: utf-8 -*-
import json
from datasets import Dataset, DatasetDict
import pdb

with open("persona_data.json", "r", encoding="utf8") as f:
    data = json.load(f)


from sklearn.model_selection import train_test_split

train, val = train_test_split(data, test_size=0.05)
print(len(train))
print(len(val))

for element in train[:]:
    try:
       if list(element['persona'].keys()) != ['이름', '나이', '직업', '성격', '외모', '비밀', '이상']:
           train.remove(element) 
       else:
           for i in ['이름', '나이', '직업', '성격', '외모', '비밀', '이상']:
                try:
                    if type(element['persona'][i]) != str :
                        train.remove(element)
                except:
                    if type(element['persona'][i]) == list :
                        train.remove(element)
                    else:
                        print(f"error1: {element}")
    
    except:
        try:
            if list(element.keys()) != ['persona', 'question', 'answer']:
                train.remove(element)
            else:
                if type(element['persona'][i]) == list :
                    train.remove(element)
                else:
                    print(f"error2: {element}")
    
        except:
            continue
    
    
for element in val[:]:
    try:
       if list(element['persona'].keys()) != ['이름', '나이', '직업', '성격', '외모', '비밀', '이상']:
           val.remove(element) 
       else:
           for i in ['이름', '나이', '직업', '성격', '외모', '비밀', '이상']:
                try:
                    if type(element['persona'][i]) != str :
                        val.remove(element)
                except:
                    if type(element['persona'][i]) == list :
                        val.remove(element)
                    else:
                        print(f"error1: {element}")
    
    except:
        try:
            if list(element.keys()) != ['persona', 'question', 'answer']:
                val.remove(element)
            else:
                if type(element['persona'][i]) == list :
                    val.remove(element)
                else:
                    print(f"error2: {element}")
    
        except:
            continue
    
        
print(len(train))
print(len(val))


train_dataset = Dataset.from_list(train)
val_dataset = Dataset.from_list(val)




print(train_dataset)
print(val_dataset)


train_dataset = train_dataset.map(
    lambda x: {'text': f"###페르소나: {x['persona']}\n\n###질문: {x['question']}\n\n###답변: {x['answer']}<|endoftext|>"}
               )
val_dataset = val_dataset.map(
    lambda x: {'text': f"###페르소나: {x['persona']}\n\n###질문: {x['question']}\n\n###답변: {x['answer']}<|endoftext|>"}
               )

dataset = DatasetDict({'train' : train_dataset, 'valid': val_dataset})

dataset.push_to_hub("yatsby/persona_chat")