import requests, os, json

SERVER_ENDPOINT = 'https://hf.space/embed/akhaliq/Real-Time-Voice-Cloning/+/api/predict'

def voice_cloning(text, in_file, out_file):
  '''
  The text to be converted to speech
  The file to be converted from (.mp3, .wav)
  The file to be converted to (.wav)
  '''
  # files = {
  #   'in_file': (in_file, open(in_file, 'rb')),
  #   'out_file': (out_file, open(out_file, 'rb'))
  # }
  data = {
    'text': text,
    'is_example': False,
    'name': 'test.wav',
    'in_file': 'input.wav',
    'out_file': 'output.wav'
  }

  r = requests.post(SERVER_ENDPOINT, data=data)
  print(r.json())

if __name__ == "__main__":
  voice_cloning('This is crazy', 'input.wav', 'output.wav')
