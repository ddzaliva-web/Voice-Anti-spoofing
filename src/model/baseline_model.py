import torch
from torch import nn
from torch.nn import Sequential
import torchaudio
#torchaudio.set_audio_backend("sox_io")
torchaudio.set_audio_backend("soundfile")

class MFM(nn.Module):
    #forward преобразовывет входные данные в выходные
    def forward(self,tens):
        half = tens.size(1) // 2
        left = tens[:,:half]
        right = tens[:,half:]
        return torch.max(left,right)

class LCNN(nn.Module):
    """
    Large Convolutional Neural Network
    """
    def __init__(self):
        super().__init__()
        #преобразование аудио сигнала в спектограмму
        self.stft = torchaudio.transforms.Spectrogram(
                        n_fft=512,
                        win_length=512,
                        hop_length=256,
                        power=None)
        #CNN-9 model
        self.net = Sequential(
            # people say it can approximate any function...
            nn.Conv2d(1,96, kernel_size=(5,5), stride=1, padding=2),#на входе спектограмма с 1 каналом
            MFM(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(48,96, kernel_size=(1,1), stride=1, padding=0),
            MFM(),
            nn.Conv2d(48,192, kernel_size=(3,3), stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(96, 192, kernel_size=(1,1), stride=1, padding=0),
            MFM(),
            nn.Conv2d(96,384, kernel_size=(3,3), stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=(1,1), stride=1, padding=0),
            MFM(),
            nn.Conv2d(192,256, kernel_size=(3,3), stride=1, padding=1),
            MFM(),
            nn.Conv2d(128, 256, kernel_size=(1,1), stride=1, padding=0),
            MFM(),
            nn.Conv2d(128,256, kernel_size=(3,3), stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((8,8)),#приведение всех картинок(спектограмм) к 1 размеру
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            MFM(),
            nn.Dropout(0.6), #регуляризация которая выключает 60% нейронов дабы не переобучаться
            nn.BatchNorm1d(256), #нормализует выходные нейроны, делая равномерное распределение (для ускорения обучения)
            #этот вектор 256 описывает аудио для задачи подделка или оригинал
            nn.Linear(256,2), #превращает вектор размера 512 в 2 числа(0,1)


    )

    def forward(self, data_object, **kwargs):
       spec = self.stft(data_object).abs() #берем магнитуду
       spec = torch.log(spec.clamp(min=1e-9))
       spec = spec.unsqueeze(1) #conv2d ожидает (batch,1(channels),freq,time)

       return {"logits" : self.net(spec)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
