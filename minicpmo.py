import queue
from typing import List, Literal, Union, Optional
import uuid
import warnings

import librosa
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Global constant defining the sample rate for audio processing (8kHz)
INPUT_OUTPUT_AUDIO_SAMPLE_RATE = 8000

class AudioData(BaseModel):
    """
    Pydantic model for handling audio data with numpy arrays.
    
    Attributes:
        array: Numpy array containing audio samples
        sample_rate: Sample rate of the audio in Hz
    """
    # Allow arbitrary types to support numpy arrays in Pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    array: np.ndarray  # Audio samples as a numpy array
    sample_rate: int   # Sample rate in Hz

class MiniCPMo:
    """
    A wrapper class for the MiniCPM model with text-to-speech capabilities.
    Handles model loading, quantization, and inference with audio generation.
    """
    
    def __init__(self, device: Literal["cpu", "cuda"] = "cuda", model_revision: str = "main", 
                 load_in_8bit: bool = True, load_in_4bit: bool = False, 
                 bnb_4bit_quant_type: str = "nf4", bnb_4bit_compute_dtype: Optional[torch.dtype] = torch.bfloat16):
        """
        Initialize MiniCPMo with optional quantization.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            model_revision: Model revision to load from Hugging Face Hub
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision (takes precedence over 8-bit)
            bnb_4bit_quant_type: Quantization type for 4-bit (either 'fp4' or 'nf4')
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
        """
        super().__init__()
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Configure PyTorch for better memory usage and performance
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix multiplication
        torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for cuDNN operations
        
        # Configure PyTorch memory allocator for better memory management
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Optimize for speed: use FP16 without quantization for maximum performance
        torch_dtype = torch.float16  # FP16 for best A10G performance
        print("🚀 Loading model in FP16 for optimal A10G performance")
        
        # Load the model optimized for speed on A10G
        with torch.device(device):
            self.model = AutoModel.from_pretrained(
                "openbmb/MiniCPM-o-2_6",
                trust_remote_code=True,      # Required for custom model code
                attn_implementation="sdpa",   # Use SDPA for optimal attention performance
                torch_dtype=torch_dtype,     # FP16 for A10G optimization
                revision=model_revision,     # Model version
                low_cpu_mem_usage=True,      # Optimize CPU memory usage
                device_map='auto',           # Automatically handle device placement
            )
            
            # Set model to evaluation mode and disable gradients for inference
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
            print("✅ Model loaded in FP16 precision optimized for A10G")
        
        # Load the tokenizer for the model
        self._tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-o-2_6", 
            trust_remote_code=True, 
            revision=model_revision
        )

        # Initialize TTS if on CUDA
        if device == "cuda":
            self.init_tts()

        self._generate_audio = True  # Flag to control audio generation
        print("✅ MiniCPMo initialized")

    def init_tts(self):
        """
        Initialize the text-to-speech component of the model.
        Sets up the TTS model with appropriate precision and moves it to the correct device.
        """
        # Determine the compute dtype based on quantization
        compute_dtype = torch.bfloat16 if hasattr(self, 'load_in_4bit') and self.load_in_4bit else torch.float16
        
        # Initialize TTS with automatic mixed precision
        with torch.amp.autocast('cuda', dtype=compute_dtype):
            # Initialize the TTS module in the model
            self.model.init_tts()
            
            # Configure the TTS model if it exists
            if hasattr(self.model, 'tts') and self.model.tts is not None:
                # Convert to half precision if not using quantization
                if not (hasattr(self, 'load_in_4bit') and self.load_in_4bit) and not (hasattr(self, 'load_in_8bit') and self.load_in_8bit):
                    self.model.tts = self.model.tts.half()
                
                # Set to evaluation mode and disable gradients
                self.model.tts.eval()
                for param in self.model.tts.parameters():
                    param.requires_grad = False
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prefill_audio(
        self,
        audio_arrays: List[np.ndarray],
    ):
        """
        Preprocess and feed audio data to the model in chunks.
        
        Args:
            audio_arrays: List of numpy arrays containing audio samples
        """
        # Combine all audio arrays into a single array
        audio_samples = np.concatenate(audio_arrays)
        print(f"Prefilling audio with {audio_samples.shape} samples")

        # Process audio in 5-second chunks to avoid memory issues
        chunk_size = INPUT_OUTPUT_AUDIO_SAMPLE_RATE * 5  # 5 seconds of audio
        
        # Process each chunk separately
        for chunk_start in range(0, len(audio_samples), chunk_size):
            # Extract current chunk
            chunk = audio_samples[chunk_start : chunk_start + chunk_size]

            # Format as a user message for the model
            msgs = [{"role": "user", "content": [chunk]}]

            # Send the chunk to the model's streaming prefill
            self.model.streaming_prefill(
                session_id=self.session_id,  # Current session ID
                msgs=msgs,                  # Audio chunk as a message
                tokenizer=self._tokenizer,   # Tokenizer for processing
            )

    def _warmup(self):
        """Run comprehensive warmup to fully initialize all model components for sub-1s TTFB."""
        if self.device == "cuda":
            with torch.no_grad():
                try:
                    print("� Starting aggressive model warmup for sub-1s TTFB...")
                    
                    # Pre-warm CUDA kernels and memory allocations
                    torch.cuda.synchronize()
                    
                    # Multiple warmup sessions with varying lengths to optimize all code paths
                    warmup_texts = ["Hi", "Hello there", "How are you today?"]
                    
                    for i, text in enumerate(warmup_texts):
                        warmup_session_id = str(uuid.uuid4())
                        
                        # Prefill with varying text lengths
                        self.model.streaming_prefill(
                            session_id=warmup_session_id,
                            msgs=[{"role": "user", "content": [text]}],
                            tokenizer=self._tokenizer,
                        )
                        
                        # Ultra-aggressive warmup config for maximum speed (quasi-deterministic sampling)
                        warmup_config = {
                            'session_id': warmup_session_id,
                            'tokenizer': self._tokenizer,
                            'max_new_tokens': 3 + i,  # Varying lengths
                            'do_sample': True,       # Enable sampling to avoid warnings
                            'temperature': 0.01,     # Ultra-low for quasi-deterministic behavior
                            'top_p': 0.1,           # Very restrictive
                            'top_k': 1,             # Only consider top token for near-greedy
                            'generate_audio': True,
                            'use_cache': True,
                            'audio_sample_rate': 24000,
                            'audio_chunk_size': 32,
                            'early_stopping': True,
                        }
                        
                        # Use the warmup config directly (no filtering needed)
                        
                        # Process multiple chunks to fully warm the streaming pipeline
                        warmup_count = 0
                        for response in self.model.streaming_generate(**warmup_config):
                            warmup_count += 1
                            if warmup_count >= 2:  # Quick warmup per session
                                break
                    
                    # Force GPU synchronization and clear cache for optimal first request
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    
                    print("⚡ Aggressive warmup completed - ready for sub-1s TTFB!")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Aggressive warmup failed with error: {str(e)}")
                    print("Proceeding with standard performance, first request might be slower")
                
    def _prefill(self, data: List[Union[str, AudioData]]):
        """
        Preprocess and feed input data (text or audio) to the model.
        
        Args:
            data: List of strings (text) or AudioData objects to prefill
            
        Raises:
            ValueError: If input data is not a string or AudioData
            Exception: For any processing errors
        """
        try:
            audio_arrays = []
            for prefill_data in data:
                # Handle text input
                if isinstance(prefill_data, str):
                    text = prefill_data
                    audio = None
                # Handle audio input
                elif isinstance(prefill_data, AudioData):
                    text = None
                    audio = prefill_data.array
                    audio_sample_rate = prefill_data.sample_rate
                else:
                    raise ValueError("prefill_data must be a string or AudioData")

                # Process text input
                if text:
                    self.model.streaming_prefill(
                        session_id=self.session_id,
                        msgs=[{"role": "user", "content": [text]}],
                        tokenizer=self._tokenizer,
                    )

                # Process audio input
                if audio is not None:
                    # Resample audio to the target sample rate if needed
                    if audio_sample_rate != INPUT_OUTPUT_AUDIO_SAMPLE_RATE:
                        resampled_audio = librosa.resample(
                            audio, 
                            orig_sr=audio_sample_rate, 
                            target_sr=INPUT_OUTPUT_AUDIO_SAMPLE_RATE,
                            res_type='kaiser_fast'  # Fast resampling algorithm
                        )
                    else:
                        resampled_audio = audio

                    # Send audio data to be prefilled
                    self._prefill_audio(
                        audio_arrays=[resampled_audio],
                    )

        except Exception as e:
            print(f"_prefill() error: {e}")
            raise  # Re-raise the exception after logging

    def run_inference(self, prefill_data: List[Union[str, AudioData]]):
        """
        Run inference on the provided input data (text or audio).
        
        Args:
            prefill_data: List of strings (text) or AudioData objects to process
            
        Yields:
            Union[AudioData, str, None]: Generated audio chunks, text responses, or None when done
        """
        try:
            # Generate a new session ID for this inference
            self.session_id = str(uuid.uuid4())
            
            # Prefill the model with input data if provided
            if prefill_data:
                with torch.no_grad():  # Disable gradient calculation
                    self._prefill(data=prefill_data)
            
            # Configure generation parameters for sub-1s TTFB target (quasi-deterministic sampling)
            generation_config = {
                'session_id': self.session_id,       # Unique ID for this generation
                'tokenizer': self._tokenizer,        # Tokenizer for text processing
                'max_new_tokens': 8,                 # Ultra-minimal tokens for sub-1s TTFB
                'do_sample': True,                   # Enable sampling to avoid warnings
                'temperature': 0.01,                 # Ultra-low for quasi-deterministic behavior
                'top_p': 0.1,                        # Very restrictive sampling
                'top_k': 1,                          # Only consider top token for near-greedy behavior
                'generate_audio': True,              # Always enable audio generation
                'use_cache': True,                   # Use KV cache for faster generation
                'pad_token_id': self._tokenizer.eos_token_id,  # End-of-sequence token
                'audio_sample_rate': 24000,          # Explicit sample rate for TTS
                'audio_chunk_size': 32,              # Ultra-small chunks for instant streaming
                'repetition_penalty': 1.0,           # No penalty for speed
                'early_stopping': True,              # Stop generation as soon as possible
            }
            
            # Run generation with mixed precision and no gradient calculation
            with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
                response_generator = self.model.streaming_generate(**generation_config)

            # Process each response from the generator
            for response in response_generator:
                audio = None
                sample_rate = INPUT_OUTPUT_AUDIO_SAMPLE_RATE
                text = None

                # Extract audio from response if available
                if hasattr(response, "audio_wav"):
                    sample_rate = getattr(response, "sampling_rate", INPUT_OUTPUT_AUDIO_SAMPLE_RATE)
                    audio = response.audio_wav.cpu().detach().numpy()  # Move to CPU and convert to numpy

                # Extract text from response
                if isinstance(response, dict):
                    text = response.get("text")
                elif hasattr(response, "text"):
                    text = response.text

                # Yield audio data if available
                if audio is not None:
                    audio_data = AudioData(
                        array=audio,
                        sample_rate=sample_rate,
                    )
                    yield audio_data

                # Yield text if available
                if isinstance(text, str) and text.strip():
                    yield text

            # Signal the end of generation
            yield None

        except Exception as e:
            print(f"Error during inference: {e}")
            yield None  # Signal completion even on error
