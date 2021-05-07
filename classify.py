from datetime import date
from models import AudioClassifier
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AudioData, TensorBoard, Validation
from sklearn.model_selection import KFold

def get_audio_files(path):
	files = []
	for i in range(60):
		speaker = i+1
		if speaker < 10:
			speaker = f'0{speaker}'
		else:
			speaker = f'{speaker}'

		new_path = f'{path}{speaker}/'
		new_files = [f'{speaker}/{f}' for f in os.listdir(new_path)]

		files += new_files

	df = pd.DataFrame({'relative_path':files,'classID':[int(f.split('/')[-1].split('_')[0]) for f in files]})

	return df

def training(model, train_dl, device, num_epochs=1, save_path='models/model.pt', writer=None):
	# Loss, Optimizer, Scheduler
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001
													,steps_per_epoch = int(len(train_dl))
													,epochs=num_epochs
													,anneal_strategy='linear')

	for epoch in range(num_epochs):
		running_loss = 0.0
		correct_prediction = 0
		total_prediction = 0

		for i, data in tqdm(enumerate(train_dl)):
			inputs, labels = data[0].to(device), data[1].to(device)

			inputs_m, inputs_s = inputs.mean(), inputs.std()
			inputs = (inputs - inputs_m) / inputs_s

			optimizer.zero_grad()

			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			scheduler.step()

			running_loss += loss.item()

			_, prediction = torch.max(outputs, 1)
			correct_prediction += (prediction == labels).sum().item()
			total_prediction += prediction.shape[0]

			if writer and i % 100 == 99:
				writer.add_scalar('training loss',
					running_loss / 1000,
					epoch * len(train_dl) + i)

				writer.add_figure('predictions vs. actuals',
					TensorBoard.plot_classes_preds(model, inputs, labels),
					global_step=epoch * len(train_dl) + i)

		if epoch+1 % 20 == 19:
			torch.save({
				'epoch':num_epochs,
				'model_state_dict':model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'loss':loss,
				}, f'models/checkpoint_epoch{epoch+1}.pt')

		num_batches = len(train_dl)
		avg_loss = running_loss / num_batches
		acc = correct_prediction / total_prediction
		print(f'Epoch: {epoch+1}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

	save_model(model, optimizer, num_epochs, avg_loss, f'model_{num_epochs}epochs_{date.today().strftime("%m-%d-%y")}.pt')
	
	print('Training Complete.')

def testing(model, test_dl, device, cm_path='figures/confusion.png'):
	correct_prediction = 0
	total_prediction = 0
	y_true = []
	y_pred = []

	with torch.no_grad():
		for data in test_dl:
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

def reset_weights(model):
	for layer in model.children():
		if hasattr(layer, 'reset_parameters'):
			print(f'Reset trainable parameters of layer = {layer}')
			layer.reset_parameters()

def kfold_cross_validation(data, folds=5, epochs=10, writer=None):
	accuracies = []
	kfold = KFold(n_splits=folds, shuffle=True)

	for fold, (train, test) in enumerate(kfold.split(data)):
		train_subsampler = torch.utils.data.SubsetRandomSampler(train)
		test_subsampler = torch.utils.data.SubsetRandomSampler(test)
		
		train_dl = torch.utils.data.DataLoader(data, batch_size=16, sampler=train_subsampler)
		test_dl = torch.utils.data.DataLoader(data, batch_size=16, sampler=test_subsampler)

		model = AudioClassifier()
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model = model.to(device)
		next(model.parameters()).device
		model.apply(reset_weights)

		training(model, train_dl, num_epochs=epochs, device=device, writer=writer)

		acc = testing(model, test_dl, device, f'figures/confusion_fold{fold}_{date.today().strftime("%m-%d-%y")}.png')
		accuracies.append(acc)

	print([f'{v:.2f} accuracy for fold {k+1}' for k,v in enumerate(accuracies)])

def complete_training(data, epochs=10, train_size=0.8, writer=None):
	num_items = len(data)
	num_train = round(num_items * train_size)
	num_test = num_items - num_train
	train_data, test_data = random_split(data, [num_train, num_test])

	train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
	test_dl = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

	model = AudioClassifier()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	next(model.parameters()).device

	training(model, train_dl, device, epochs, writer=writer)
	testing(model, test_dl, device, f'figures/confusion_{date.today().strftime("%m-%d-%y")}.png')

def clear_checkpoints(path):
	for file in os.listdir(path):
		if 'checkpoint' in file:
			os.remove(path+file)

def save_model(model, optimizer, epochs, loss, file):
	torch.save({
		'epoch':epochs,
		'model_state_dict':model.state_dict(),
		'optimizer_state_dict':optimizer.state_dict(),
		'loss':loss,
		}, f'models/{file}')

def load_model(file, train=False):
	model = AudioClassifier()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	checkpoint = torch.load('./models/'+file)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	if train:
		model.train()
	else:
		model.eval()

	return model

if __name__ == '__main__':
	data_path = 'F:/audio/audio_mnist/data/'
	df = get_audio_files(data_path)
	data = AudioData(df, data_path)
	writer = SummaryWriter('runs/audio_mnist_run_test')
	clear_checkpoints('./models/')

	# Comment/Uncomment one of the below to run 1 full training or k-fold cross val
	#kfold_cross_validation(data, folds=5, epochs=10)
	complete_training(data, epochs=10, writer=None)