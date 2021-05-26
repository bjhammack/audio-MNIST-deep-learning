from datetime import date
from get_keys import key_check
from models import AudioClassifier
import numpy as np
import os
import pandas as pd
from scipy.io.wavfile import write
import sounddevice as sd
from time import sleep
import torch
from utils import AudioData

def record_audio():
	sr = 48000
	seconds = 1.5

	recording = sd.rec(int(seconds * sr), samplerate=sr, channels=2)
	sd.wait()
	write('./data/live_audio/audio.wav', sr, recording)

def load_model(path):
	model = AudioClassifier()
	model.load_state_dict(torch.load(f'./models/{path}'))
	model.eval()

	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = torch.device('cpu')
	model = model.to(device)
	next(model.parameters()).device

	return model, device

def predict(model, dl, device):
	correct_prediction = 0
	total_prediction = 0
	y_true = []
	y_pred = []

	with torch.no_grad():
		for data in dl:
			input = data[0].to(device)

			output = model(input)

			prediction = np.argmax(output)#int(torch.max(output.data,1)[1].numpy())
			print(prediction)

if __name__ == '__main__':
	data_path = 'C:/Users/sourp/Documents/programming/github/venv_audio-mnist-deep-learning/audio-MNIST-deep-learning/data/live_audio/'
	model, device = load_model('final_model_10epochs_05-09-21.pt')
	quit = False
	print('Ready to record.')
	while True:
		if quit:
			break
		else:
			keys = key_check()
			if 'R' in keys:
				print('Starting recording...')
				for i in range(3,0,-1):
					print(i)
					sleep(1)
				print('Recording...')
				record_audio()
				data = AudioData(pd.DataFrame({'relative_path':['audio.wav'],'classID':[10]}), data_path, False)
				dl = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False)	
				predict(model, dl, device)
			elif 'Q' in keys:
				quit = True