import google.generativeai as genai
from collections import OrderedDict
import json

import os

GOOGLE_API_KEY = "YOUR API KEY"

genai.configure(api_key=GOOGLE_API_KEY)

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]
model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)

file_data = OrderedDict()

import random
import pandas as pd
import time

retry = 0
fail = 0
def make_data():
  for i in range(30000):
    print(i)
    try:
      try:
        response = model.generate_content("""
                                                  1. 페르소나를 만들려고 합니다. 다음 페르소나 예시를 참고해서 똑같은 형식으로 다른 페르소나를 생성해 주세요.
                                                  예시 : "이름: 루나/나이: 25세/직업: 마술사, 성직자/성격: 침착하고 냉정, 뛰어난 판단력과 지혜, 신비로운 매력, 강인한 의지/외모: 긴 은빛 머리, 하늘색 눈, 마법의 빛으로 둘러싸인 신체, 화려한 마법사 옷, 빛나는 지팡이/비밀: 선천적인 마법 능력, 엄격한 수행으로 완벽한 마법사, 중립적 입장, 미래 예지, 세상을 구할 수 있는 사람을 찾음/이상: 세상의 평화 유지, 선악 중립, 세상을 구할 수 있는 사람 찾기"
                                                  
                                                  2. 상대방에게 할 법한 질문이나 대화를 한 문장 생성해주세요. 질문이나 대화의 내용은 최대한 자세한 편이면 좋겠습니다. 상대방은 1번에서 새로 생성한 페르소나와 같은 사람입니다. 질문이나 대화에 상대방의 이름을 포함하지 않고, 한국어 외 다른 언어는 사용하지 않아주세요.
                                                  
                                                  3. 당신은 1번에서 새로 생성한 페르소나로서 2번에서 생성한 질문에 대한 대답을 해주세요. 초성체(ㅋㅋㅋ, ㅎㅎㅎ, 등등)를 사용해도 됩니다. 200자 이내로 최대한 상세하게 답변해주세요.
                                                  
                                                  당신은 1,2,3번에서 생성한 내용을 다음과 같은 JSON 형태로 정리해서 답변을 출력해주세요. JSON을 제외한 다른 문구는 출력하지 말아주세요.
                                                  {
                                                    "persona": {
                                                      "이름": {이름},
                                                      "나이": {나이},
                                                      "직업": {직업},
                                                      "성격": {성격},
                                                      "외모": {외모},
                                                      "비밀": {비밀},
                                                      "이상": {이상},
                                                      },
                                                    "question":{새로 생성한 질문},
                                                    "answer": {새로 생성한 답변}
                                                  }""").text
        # if response_persona.startswith('죄송합니다.'):
        #   print("error")
        #   continue
        data_strip1 = response.lstrip("```JSONjson\n").rstrip("\n```")
        print(data_strip1)
        try:
          response_json = json.loads(data_strip1)
          
          with open("persona_data.json", "r", encoding = "utf8") as read_file:
            data = json.load(read_file)
          data.append(response_json)
        
          with open("persona_data.json", "w", encoding="utf8") as make_file:
            json.dump(data, make_file, ensure_ascii=False, indent="\t")
        except ValueError:
          continue

      except ValueError:
        continue
    except:
      global retry
      global fail
      retry += 1
      if retry == 2:
        fail = 1
        break
      time.sleep(10)

make_data()
print(fail)




