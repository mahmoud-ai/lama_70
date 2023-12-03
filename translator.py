import requests 
import json


def translate(text, tgt_lang):
  url = 'http://184.105.208.39:4321'

  # Data you want to send in the POST request
  data = {
      'text': text ,
      'language': tgt_lang,
      'dialect': 'MarianMT'
  }

  # Make the POST request
  response = requests.post(url, json=data)

  # Check if the request was successful
  if response.status_code == 200:
      tex = response.json().get('translated_text')  # Print the response if successful
#      print((tex))
      return tex
  else:
      print("Failed with status code:", response.status_code)
      print(response.text)  # Print the error if failed

print(translate("واسحب بساطك من بلاط الحاكم العربي حتي لا يعلقها وساما", tgt_lang="english"))
