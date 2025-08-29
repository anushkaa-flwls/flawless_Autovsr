import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# import whisper
# from whisper.audio import log_mel_spectrogram

from transformers import WhisperProcessor, WhisperModel
import torchaudio

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E




class MultilingualAudioBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_fps = 25
        self.feat_fps = 50
        self.sr = 16000
        self.model_name = "autoavsr"
        print("--------------------------------")
        print(f"audio backbone model_name: {self.model_name}")
        print("--------------------------------")
        # Load processor and model based on model type
        if "autoavsr" in self.model_name:
            
            self.model =  E2E(odim=5049, modality="audio")
        # elif "mHuBERT" in model_name or "hubert" in model_name:
        #     self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        #     self.model = HubertModel.from_pretrained(model_name, output_hidden_states=True)
        # elif "contentvec" in model_name:
        #     self.procssor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        #     self.model = HubertModelWithFinalProj.from_pretrained(model_name, output_hidden_states=True)
        # elif "whisper" in model_name:
        elif "whisper" in self.model_name:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperModel.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if hasattr(self, 'model') and hasattr(self.model, 'eval'):
            self.model.eval()
        
       
    def forward(self, audio: torch.Tensor, frame_num=None):
        """
        
        
        whisper is configured at 50FPS
        audio: torch.Tensor of shape (batch, time) or (time,)
        sampling_rate: int, default 16kHz
        """
        
        
        if self.model_name != "autoavsr":
    
            if isinstance(audio, torch.Tensor) and audio.ndim == 2:
                audio = [a.squeeze().numpy() for a in audio]  # Convert batch to list of 1D NumPy arrays
            elif isinstance(audio, torch.Tensor) and audio.ndim == 1:
                audio = [audio.squeeze().numpy()]


        
        #extra padding for 30 ms
        
        
        
        if self.model_name == "autoavsr":
            with torch.no_grad():
                audio = audio.unsqueeze(2)
                B, T , N = audio.shape 
                print(audio.shape , "Audio shape")
 
                ilens = torch.full((B,), T, dtype=torch.long) #generating mask for attention

                outputs = self.model(audio , ilens, audio_embeddings=None)
  
        return outputs












model = MultilingualAudioBackbone()
dummy_audio = torch.randn(4, 16000 * 4)  # (B, T)

# Run forward
output = model(dummy_audio) #torch.Size([4, 150, 768]) output shape
print(output.shape , "output shape")














