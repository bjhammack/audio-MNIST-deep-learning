# Audio-MNIST-deep-learning
Using the [audio MNIST](https://github.com/soerenab/AudioMNIST) dataset, created by Becker et al. for their paper ["Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals"](https://arxiv.org/abs/1807.03418), I perform deep learning, using a PyTorch Neural Network, to accurately identify numbers being spoken.

## Data 
The data consists of 30,000 audio samples stored in .wav files. These samples are even distributed between the numbers 0-9 and between 60 speakers. Each speaker speaks the same number for the same number of iterations as every other speaker.

## Analysis
The process for the analysis itself can be found in the Jupyter notebook [data_analysis.ipynb](https://github.com/bjhammack/audio-MNIST-deep-learning/blob/main/data_analyis.ipynb).

### The Labels
The labels were examined to confirm their distribution and, as seen in the graph below, all labels had the exact same number of samples (3,000).

![Label Distribution](https://github.com/bjhammack/audio-MNIST-deep-learning/blob/main/images/label_distribution.png?raw=true "Label Hist")

### Sample Rates
For audio modeling, sample rates are important because it lets you know how much data you get getting in one second of audio, much like resolution in images. When the data is prepared for modeling, ensuring each clip has the same sample rate is important to get accurate results.

Examing the sample rates of each sample within this dataset reveals that they all have the same sample rate of 48,000.

### Sample Lengths
Much like image modeling, all audio samples will need to be the same length to get passed into the model.

Looking at the histogram below, no the lengths are normally distributed, no less than 0.3 seconds, and no greater than 1 second. The shorter lengths will need to be padded by some method to allow modeling.

![Length Histogram](https://github.com/bjhammack/audio-MNIST-deep-learning/blob/main/images/length_distribution.png?raw=true "Length Hist")

In additional to examining the distribution, it is important to also understand how each individual label's sample lengths are distributed, to ensure the model does not overfit based on clip length.

The scatter plot below shows that, while some numbers have higher floors and others lower ceilings, the majority of clip lengths are gathered around the mean of clip lengths, which is good for modeling. To be more specific, no label has a mean clip length less than 0.55 seconds or greater than 0.73 seconds.

![Length Scatter](https://github.com/bjhammack/audio-MNIST-deep-learning/blob/main/images/length_by_label.png?raw=true "Length Scatter")

### Mel Spectrogram
Before moving to data preparation, it is good to understand how the data will look prior to our model. In the case of audio data, instead of feeding a model raw samples, it has been shown to be more effective to feed neural networks spectrograms--images of the audio--to process how you would normally model images.

Mel Spectrograms in particular are effective for audio monitoring as they apply a Fourier Transformation and then scale the audio to fit within the more visually usuable Mel Scale.

There are many libraries, including some built into PyTorch, Librosa, or scikit-learn, that can create Mel Spectrograms. The below example was made using Librosa, but PyTorch will be used for the actual modeling, as that makes a cleaner pipeline.

![Mel Spectrogram](https://github.com/bjhammack/audio-MNIST-deep-learning/blob/main/images/mel_spectrogram.png?raw=true "Mel Spec")


## Data Prep
To prepare the data, the audio samples are stored in a custom `AudioData` class, which is built on PyTorch's `Dataset` class. Once in `AudioData`, cleaning/processing functions are called from the custom class `AudioUtil` (both AudioData and AudioUtil can be found in [utils.py](https://github.com/bjhammack/audio-MNIST-deep-learning/blob/main/utils.py)).

Once the file is read in, five cleaning steps occur, based on the results of analysis, in addition to best practices for audio data preparation.
1. The sample is passed through `AudioUtil.resample()`.
	* This uses torchaudio's `Resample` function to adjust the sample's sample rate to the default sample rate I identified in analysis (48,000).
2. The sample is passed through `AudioUtil.rechannel()`.
	* Much like the variance of images and their different color schemes (RGB, GRB, etc.), audio can be created on one or two channels. The model will expect two channels, so all one channel samples are converted to two channel.
3. The sample is passed through `AudioUtil.pad_trunc()`.
	* This sets all samples to a default length (in this case 1000ms or one second), using white noise to pad.
4. The sample is then converted to a Mel Spectrogram using PyTorch's `Mel_Spectrogram()` function.
5. Finally, the spectrogram is randomly masked.
	* Random frequencies and periods of time are blocked out in each spectrogram, creating additional noise to prevent overfitting and improve generality.

## Modeling


## Results

### v1

### v2

### v3

#### K-fold Cross Validation