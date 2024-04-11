# from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import io
import soundfile as sf


class ASRManager:
    # the rough idea here is to have the model loading take place in the manager __init__
    # such that once the manager is instantiated, the model is ready to take inference requests
    def __init__(self):
        # self.cfg = "auto" if cfg['device'] == "auto" else {"":"cpu"}
        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(
            "openai/whisper-small", device_map=self.device
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-small", device_map=self.device
        )
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

    def transcribe(self, audio: dict):
        sample = audio["audio"]
        input_features = self.processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device)
        # generate token ids
        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=self.forced_decoder_ids
        )
        # decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription

    def stt(self, audio_bytes):
        data = {}
        audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
        data["audio"] = {"array": audio_data, "sampling_rate": samplerate}
        return self.transcribe(data)[0]
