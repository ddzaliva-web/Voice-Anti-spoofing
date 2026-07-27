import torch
from torch.nn.utils.rnn import pad_sequence
#с помощью pad_sequence собираем в batch последовательности разной длины,дополнив их нулями до максмальной длины
def collate_fn(dataset_items):
    wavs = []
    labels = []    

    for item in dataset_items:
        wavs.append(item["data_object"])
        labels.append(item["labels"])


    wavs = pad_sequence(wavs, batch_first=True)    
    labels = torch.Tensor(labels).long() #int64
    return {"data_object" : wavs, "labels" : labels}
