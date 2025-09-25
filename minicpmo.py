import queue
from typing import List, Literal, Union
import uuid

import librosa
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from transformers import AutoModel, AutoTokenizer


INPUT_OUTPUT_AUDIO_SAMPLE_RATE = 16000

class AudioData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    array: np.ndarray
    sample_rate: int

class MiniCPMo:
    def __init__(self, device: Literal["cpu", "cuda"] = "cuda", model_revision: str = "main"):
        super().__init__()
        self.device = device


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
       import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        
       
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.device(device):
            self.model = AutoModel.from_pretrained(
                "openbmb/MiniCPM-o-2_6",
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=torch.float16,
                revision=model_revision,
                low_cpu_mem_usage=True,
                device_map='auto',
                offload_folder='offload',
                offload_state_dict=True
            )
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        self._tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-o-2_6", trust_remote_code=True, revision=model_revision
        )

        if device == "cuda":
            self.init_tts()

        self._generate_audio = True
        print("✅ MiniCPMo initialized")

    def init_tts(self):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            self.model.init_tts()
            if hasattr(self.model, 'tts') and self.model.tts is not None:
                self.model.tts.half()
                self.model.tts.eval()
                for param in self.model.tts.parameters():
                    param.requires_grad = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prefill_audio(
        self,
        audio_arrays: List[np.ndarray],
    ):
        audio_samples = np.concatenate(audio_arrays)
        print(f"prefilling audio with {audio_samples.shape} samples")

        chunk_size = INPUT_OUTPUT_AUDIO_SAMPLE_RATE * 5
        for chunk_start in range(0, len(audio_samples), chunk_size):
            chunk = audio_samples[chunk_start : chunk_start + chunk_size]

            msgs = [{"role": "user", "content": [chunk]}]

            self.model.streaming_prefill(
                session_id=self.session_id,
                msgs=msgs,
                tokenizer=self._tokenizer,
            )

    def _prefill(self, data: List[str | AudioData]):
        try:
            audio_arrays = []
            for prefill_data in data:
                if isinstance(prefill_data, str):
                    text = prefill_data
                    audio = None
                elif isinstance(prefill_data, AudioData):
                    text = None
                    audio = prefill_data.array
                else:
                    raise ValueError(f"._prefill(): prefill_data must be a string or AudioData")

                if text:
                    self.model.streaming_prefill(
                        session_id=self.session_id,
                        msgs=[{"role": "user", "content": [text]}],
                        tokenizer=self._tokenizer,
                    )

                if audio is not None:
                    resampled_audio = librosa.resample(
                        audio, 
                        orig_sr=audio.sample_rate, 
                        target_sr=INPUT_OUTPUT_AUDIO_SAMPLE_RATE,
                    )

                    self._prefill_audio(
                        audio_arrays=[resampled_audio],
                    )

        except Exception as e:
            print(f"_prefill() error: {e}")
            raise e

    def run_inference(self, prefill_data: List[str | AudioData]):
        print("MiniCPMo _run_inference() function called")
        
      
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model.eval()
        
        try:
            self.session_id = str(uuid.uuid4())
            
       
            if prefill_data:
                with torch.no_grad():
                    self._prefill(data=prefill_data)
            
          
            generation_config = {
                'session_id': self.session_id,
                'tokenizer': self._tokenizer,
                'temperature': 0.1,  
                'top_p': 0.9,       
                'top_k': 50,        
                'max_new_tokens': 50, 
                'do_sample': True,   
                'generate_audio': self._generate_audio,
                'use_cache': True,  
                'pad_token_id': self._tokenizer.eos_token_id,
            }
            
            with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
                response_generator = self.model.streaming_generate(**generation_config)

            for response in response_generator:
                audio = None
                sample_rate = INPUT_OUTPUT_AUDIO_SAMPLE_RATE
                text = None

                # extract audio from response
                if hasattr(response, "audio_wav"):
                    has_audio = True
                    sample_rate = getattr(response, "sampling_rate", INPUT_OUTPUT_AUDIO_SAMPLE_RATE)
                    audio = response.audio_wav.cpu().detach().numpy()

                # check for text
                if isinstance(response, dict):
                    text = response.get("text")
                elif hasattr(response, "text"):
                    text = response.text

                # put audio in output queue
                if audio is not None:
                    audio_data = AudioData(
                        array=audio,
                        sample_rate=sample_rate,
                    )

                    yield audio_data

                # put text in output queue
                if isinstance(text, str) and text:
                    has_text = True
                    yield text

            yield None

        except Exception as e:
            print(f"_run_inference() error: {e}")
            yield None
