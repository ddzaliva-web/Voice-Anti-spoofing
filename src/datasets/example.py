import torch
from tqdm.auto import tqdm
from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
import torchaudio
DATA_ROOT = Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA")
#torchaudio.set_audio_backend("sox_io")
torchaudio.set_audio_backend("soundfile")

class  ASVSpoofDataset(BaseDataset):
    """
   ASVSpoofDataset
    """

    def __init__(self,name="train", *args, **kwargs):
        """
        Args:
              name (str): "train", "dev", "eval".
        """
        index = self._create_index(name)
        super().__init__(index, *args, **kwargs)

    def _create_index(self,name):
        protocol_files = {
                    "train" : "ASVspoof2019.LA.cm.train.trn.txt",
                    "dev" : "ASVspoof2019.LA.cm.dev.trl.txt",
                    "eval" : "ASVspoof2019.LA.cm.eval.trl.txt"
                }
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): "train", "dev", "eval".
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = DATA_ROOT / "ASVspoof2019_LA_cm_protocols" / protocol_files[name]
        # to get pretty object names
        # In this example, we create a synthesized dataset. However, in real
        # tasks, you should process dataset metadata and append it
        # to index. See other branches.

        with open(data_path, 'r') as f:
            for line in f:
                audio_line = line.strip().split()
                audio_name = audio_line[1]
                label = audio_line[-1]
                if label == 'bonafide':
                    label = 1
                else:
                    label = 0
                audio_path = DATA_ROOT / f"ASVspoof2019_LA_{name}" / "flac" / f"{audio_name}.flac"
                index.append({"path": str(audio_path), "label": label})

        return index
    def load_object(self,path):
            audio,sr = torchaudio.load(path)
            audio = audio.squeeze(0)
            return 
