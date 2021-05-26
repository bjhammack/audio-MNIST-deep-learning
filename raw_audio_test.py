from datetime import date
from models import AudioClassifier
import os
import pandas as pd
import torch
from utils import AudioData, Validation

def get_audio_files(path):
	number_dict = {'zero.wav':0
				  ,'one.wav':1
				  ,'two.wav':2
				  ,'three.wav':3
				  ,'four.wav':4
				  ,'five.wav':5
				  ,'six.wav':6
				  ,'seven.wav':7
				  ,'eight.wav':8
				  ,'nine.wav':9
				  }
	files = [f'{f}' for f in os.listdir(path)]

	df = pd.DataFrame({'relative_path':files,'classID':[number_dict[i] for i in files]})

	return df

def load_model(path):
	model = AudioClassifier()
	model.load_state_dict(torch.load(f'./models/{path}'))
	model.eval()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	next(model.parameters()).device

	return model, device

def evaluate(model, dl, device, cm_path='figures/confusion.png'):
	correct_prediction = 0
	total_prediction = 0
	y_true = []
	y_pred = []

	with torch.no_grad():
		for data in dl:
			inputs, labels = data[0].to(device), data[1].to(device)

			inputs_m, inputs_s = inputs.mean(), inputs.std()
			inputs = (inputs - inputs_m) / inputs_s

			outputs = model(inputs)

			_, prediction = torch.max(outputs,1)
			correct_prediction += (prediction == labels).sum().item()
			total_prediction += prediction.shape[0]
			y_pred.extend(prediction.cpu())
			y_true.extend(labels.cpu())

	Validation.confusion(y_pred, y_true, cm_path)
	acc = correct_prediction/total_prediction
	print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

	return acc

if __name__ == '__main__':
	data_path = 'C:/Users/sourp/Documents/programming/github/venv_audio-mnist-deep-learning/audio-MNIST-deep-learning/data/new_audio/'
	df = get_audio_files(data_path)
	data = AudioData(df, data_path, False)
	dl = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False)
	model, device = load_model('final_model_10epochs_05-09-21.pt')

	evaluate(model, dl, device, f'figures/eval_confusion_{date.today().strftime("%m-%d-%y")}.png')