# Imports
import os

import PIL.Image
import opensoundscape
import torch
import numpy as np
from django.core.files.storage import FileSystemStorage
from torchvision import models, transforms
import librosa
import librosa.display
from PIL import Image
# from opensoundscape.audio import Audio as Au
# from opensoundscape.spectrogram import Spectrogram as Sp
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from GenreFinder import settings

print(0)


class LeNet5(torch.nn.Module):
    def __init__(self,
                 conv_size=5,
                 use_batch_norm=False):
        super(LeNet5, self).__init__()

        self.conv_size = conv_size
        self.use_batch_norm = use_batch_norm

        activation_function = torch.nn.ReLU()

        pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # shape=[3,512, 256]
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, padding=2)
        # shape=[6,512, 256]
        self.act1 = activation_function
        self.bn1 = torch.nn.BatchNorm2d(num_features=6)
        self.pool1 = pooling_layer
        # shape=[6,256, 128]

        self.conv2 = self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=2)
        # shape=[16,256, 128]
        self.act2 = activation_function
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.pool2 = pooling_layer
        # shape=[16,128, 64]
        self.fc1 = torch.nn.Linear(72 * 108 * 16, 120)
        self.act3 = activation_function

        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = activation_function

        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        if self.use_batch_norm:
            x = self.bn1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x

    def predict(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.pool2(x)

        x = x.view(x.size(0) * x.size(1) * x.size(2))
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x

    def load_state(self, path=None):
        if path is None:
            path = 'fileLoader/params.py'
        self.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, path)))
        self.eval()

    def get_predicted_genre(self, input_tensor):
        print(input_tensor.shape)
        label_dict = [
            'jazz',
            'reggae',
            'rock',
            'blues',
            'hiphop',
            'country',
            'metal',
            'classical',
            'disco',
            'pop',
        ]
        print(self.predict(input_tensor))
        prediction = label_dict[torch.argmax(self.predict(input_tensor))]
        return prediction


class File(object):
    name: str
    extension: str

    def __init__(self, name, root):
        self.path = os.path.join(root, name)
        self.name = name
        self.extension = self.name.split(".")[-1]

    def get_size(self):
        size = os.path.getsize(self.path)
        return size


class Audio(object):
    duration: float

    def __init__(self, data, sr):
        self.data = data
        self.sample_rate = sr
        self.duration = len(data) // sr

    def get_middle(self, length):
        if length > self.duration / 2:
            return False
        mid = self.duration // 2
        sr = self.sample_rate
        new_data = self.data[(mid - length//2) * sr: (mid + length//2) * sr]
        return new_data


class Spectrogram(object):
    image: PIL.Image

    def __init__(self, image):
        self.image = image

    def get_tensor(self):
        convert_tensor = transforms.ToTensor()
        tensor = convert_tensor(self.image)
        return tensor

    def get_image(self):
        return self.image


class AudioController:
    @staticmethod
    def toAudio(file: str):
        y, sr = librosa.load(file)
        audio = Audio(y, sr)
        return audio

    @staticmethod
    def toSpectro(audio: Audio):

        data = audio.data
        temp = audio.get_middle(30)
        if temp is not False:
            data = temp

        window_size = 512
        window = np.hanning(window_size)
        stft = librosa.core.spectrum.stft(data, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)

        fig = plt.Figure()
        canvas = FigureCanvas(fig)

        # canvas.draw()
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
        fig.savefig(os.path.join(settings.BASE_DIR, 'fileLoader/temp_fig.png'))
        image = Image.frombytes('RGB',[288,432], fig.canvas.tostring_rgb())
        print(image)
        spectrogram = Spectrogram(image)
        return spectrogram


class FileUploadController:
    @staticmethod
    def check_size(file: File):
        size = file.get_size()
        return size <= 10 * 1024 * 1024  # 10 MB

    @staticmethod
    def check_extension(file: File):
        extension = file.extension
        return extension == "mp3" or extension == "wav"

    @staticmethod
    def save_file(file: File, upload):
        fs = FileSystemStorage()
        fs.save(file.name, upload)
