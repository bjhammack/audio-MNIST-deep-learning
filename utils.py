import math,random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio import transforms

class AudioData(Dataset):
	def __init__(self, df, data_path, train=True):
		self.train = train
		self.df = df
		self.data_path = str(data_path)
		self.duration = 1500
		self.sr = 48000
		self.channel = 2
		self.shift_pct = 0.4

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		audio_file = self.data_path + self.df.loc[idx, 'relative_path']
		class_id = self.df.loc[idx, 'classID']

		aud = AudioUtil.open(audio_file)
		reaud = AudioUtil.resample(aud, self.sr)
		rechan = AudioUtil.rechannel(reaud, self.channel)

		dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
		#shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
		sgram = AudioUtil.spectrogram(dur_aud)

		if self.train:
			sgram = AudioUtil.spec_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

		return sgram, class_id

class AudioUtil():
	'''
	Load an audio file. Return as tensor and sample rate
	'''
	@staticmethod
	def open(audio_file):
		sig, sr = torchaudio.load(audio_file)
		return (sig, sr)

	'''
	Model expects 2 channels. Convert 1 channel audio files to 2.
	'''
	@staticmethod
	def rechannel(aud, new_channel):
		sig, sr = aud

		if sig.shape[0] == new_channel:
			return aud

		if new_channel == 1:
			resig = sig[:1, :]
		else:
			resig = torch.cat([sig,sig])
		
		return (resig, sr)

	'''
	Standardize sampling rate.
	'''
	@staticmethod
	def resample(aud, newsr):
		sig, sr = aud

		if sr == newsr:
			return aud

		num_channels = sig.shape[0]
		resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
		if num_channels > 1:
			retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
			resig = torch.cat([resig, retwo])

		return (resig, newsr)

	'''
	Standardize sample length.
	'''
	@staticmethod
	def pad_trunc(aud, max_ms):
		sig, sr = aud
		num_rows, sig_len = sig.shape
		max_len = sr//1000 * max_ms

		if sig_len > max_len:
			sig = sig[:,:max_len]
		elif sig_len < max_len:
			pad_begin_len = random.randint(0,max_len - sig_len)
			pad_end_len = max_len - sig_len - pad_begin_len

			max_noise = sig.max()
			min_noise = sig.min()
			pad_begin = (max_noise-min_noise)*torch.rand((num_rows, pad_begin_len)) + min_noise
			pad_end = (max_noise-min_noise)*torch.rand((num_rows, pad_end_len)) + min_noise

			sig = torch.cat((pad_begin, sig, pad_end), 1)

		return (sig,sr)

	'''
	Shift signal left/right by some percent; wrap the end.
	'''
	@staticmethod
	def time_shift(aud, shift_limit):
		sig, sr = aud
		_, sig_len = sig.shape
		shift_amt = int(random.random() * shift_limit * sig_len)
		return (sig.roll(shift_amt), sr)

	'''
	Generate Mel Spectrogram.
	'''
	@staticmethod
	def spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None):
		sig,sr = aud
		top_db = 80

		spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
		spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
		return spec

	'''
	Augment spectrogram by masking periods of time and periods of frequency.
	'''
	@staticmethod
	def spec_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
		_, n_mels, n_steps = spec.shape
		mask_value = spec.mean()
		aug_spec = spec

		freq_mask_param = max_mask_pct * n_mels
		for _ in range(n_freq_masks):
			aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

		time_mask_param = max_mask_pct * n_steps
		for _ in range(n_time_masks):
			aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

		return aug_spec

class TensorBoard():
	'''
	Load an audio file. Return as tensor and sample rate
	'''
	@staticmethod
	def open(audio_file):
		sig, sr = torchaudio.load(audio_file)
		return (sig, sr)

	@staticmethod
	def images_to_probs(net, images):
		'''
		Generates predictions and corresponding probabilities from a trained
		network and a list of images
		'''
		output = net(images)
		# convert output probabilities to predicted class
		_, preds_tensor = torch.max(output, 1)
		preds = np.squeeze(preds_tensor.cpu().numpy())
		
		return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

	@staticmethod
	def matplotlib_imshow(img, one_channel=False):
		if one_channel:
			img = img.mean(dim=0)
		img = img / 2 + 0.5 # unnormalize
		npimg = img.cpu().numpy()
		if one_channel:
			plt.imshow(npimg, cmap="Greys")
		else:
			plt.imshow(np.transpose(npimg, (1, 2, 0)))

	@staticmethod
	def plot_classes_preds(net, images, labels):
		'''
		Generates matplotlib Figure using a trained network, along with images
		and labels from a batch, that shows the network's top prediction along
		with its probability, alongside the actual label, coloring this
		information based on whether the prediction was correct or not.
		Uses the "images_to_probs" function.
		'''
		classes = ('0','1','2','3','4','5','6','7','8','9')
		preds, probs = TensorBoard.images_to_probs(net, images)
		# plot the images in the batch, along with predicted and true labels
		fig = plt.figure(figsize=(12, 48))
		for idx in np.arange(4):
			ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
			TensorBoard.matplotlib_imshow(images[idx], one_channel=True)
			ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
				classes[preds[idx]],
				probs[idx] * 100.0,
				classes[labels[idx]]),
						color=("green" if preds[idx]==labels[idx].item() else "red"))
		
		return fig

	@staticmethod
	def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0, writer=None):
		'''
		Takes in a "class_index" from 0 to 9 and plots the corresponding
		precision-recall curve
		'''
		tensorboard_truth = test_label == class_index
		tensorboard_probs = test_probs[:, class_index]

		writer.add_pr_curve(classes[class_index],
							tensorboard_truth,
							tensorboard_probs,
							global_step=global_step)
		writer.close()

class Validation():
	@staticmethod
	def confusion(y_pred, y_true, save_filepath='confusion.png'):
		'''
		Creates and returns confusion matrix
		'''
		classes = ('0','1','2','3','4','5','6','7','8','9')
		cf_matrix = confusion_matrix(y_true, y_pred)
		df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index=[i for i in classes]
			, columns=[i for i in classes])

		plt.close('all')
		plt.figure(figsize=(12,7))
		sn.heatmap(df_cm, annot=True)
		plt.savefig(save_filepath)
