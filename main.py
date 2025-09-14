from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf


class UrbanAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(UrbanAudio, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.flatten = nn.Flatten()

        self.second = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = 22050
transform = transforms.MelSpectrogram(
    sample_rate=sr,
    n_mels=64
)


genres = torch.load('labels.pth')

model = UrbanAudio()
model.load_state_dict((torch.load('urban_model.pth', map_location=device)))
model.to(device)
model.eval()

max_len = 500

def change_audio(waveform, sample_rate):
    if sample_rate != sr:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        # waveform = resample(waveform)
        waveform = resample(torch.tensor(waveform).unsqueeze(0))

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    elif spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))

    return spec


music_genre_app = FastAPI(title='Genre of music')


@music_genre_app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Empty file')
        waveform, sample_rate = sf.read(io.BytesIO(data), dtype='float32')
        waveform = torch.tensor(waveform).T

        spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_idx = torch.argmax(y_pred, dim=1).item()
            predicted_class = genres[pred_idx]

        return {f'Index: {pred_idx}, Sound: {predicted_class}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error: {e}')

if __name__ == '__main__':
    uvicorn.run(music_genre_app, host='127.0.0.1', port=8000)