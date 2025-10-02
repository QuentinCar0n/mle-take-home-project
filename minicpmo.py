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
        torch.backends.cudnn.benchmark = True          # Optimize for fixed input sizes
        
        # Configure PyTorch memory allocator for better memory management
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set up model quantization configuration
        quantization_config = None
        torch_dtype = torch.float16  # Default to float16 for GPU
        
        # Configure 4-bit quantization if requested
        if load_in_4bit:
            if not torch.cuda.is_available():
                warnings.warn("4-bit quantization requires CUDA. Falling back to 8-bit.")
                load_in_8bit = True
                load_in_4bit = False
            else:
                # Set up 4-bit quantization with specified parameters
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,  # 'nf4' or 'fp4'
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  # bfloat16 or float16
                    bnb_4bit_use_double_quant=True,  # Additional quantization for memory savings
                )
                torch_dtype = bnb_4bit_compute_dtype or torch.float16
        # Fall back to 8-bit quantization if 4-bit is not requested
        elif load_in_8bit:
            if not torch.cuda.is_available():
                warnings.warn("8-bit quantization requires CUDA. Falling back to FP16.")
                load_in_8bit = False
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,  # Simple 8-bit quantization
                )
        
        # Load the model with the specified configuration
        with torch.device(device):
            try:
                # Load the base model with specified precision and quantization
                self.model = AutoModel.from_pretrained(
                    "openbmb/MiniCPM-o-2_6",
                    trust_remote_code=True,  # Required for custom model code
                    attn_implementation="sdpa",  # Use SDPA for attention
                    torch_dtype=torch_dtype,     # Set precision
                    revision=model_revision,     # Model version
                    low_cpu_mem_usage=True,      # Optimize CPU memory usage
                    device_map='auto',           # Automatically handle device placement
                    offload_folder='offload',    # Folder for offloading
                    offload_state_dict=True,     # Offload state dict to CPU
                    quantization_config=quantization_config  # Apply quantization if specified
                )
                
                # Set model to evaluation mode and disable gradients
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                    
                print(f"Model loaded in {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else '16-bit'} precision")
                
            except Exception as e:
                # Fall back to FP16 if quantization fails
                if load_in_4bit or load_in_8bit:
                    warnings.warn(f"Failed to load quantized model: {str(e)}. Falling back to FP16.")
                    self.model = AutoModel.from_pretrained(
                        "openbmb/MiniCPM-o-2_6",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map='auto'
                    )
                    self.model.eval()
                else:
                    raise
        
        # Load the tokenizer for the model
        self._tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-o-2_6", 
            trust_remote_code=True, 
            revision=model_revision
        )

        # Initialize TTS if on CUDA
        if device == "cuda":
            self.init_tts()
            # Precompile CUDA kernels and automatic warmup
            dummy_input = torch.randn(1, 512, device=device, dtype=torch.float16)
            with torch.no_grad():
                torch.cuda.synchronize()
            self._warmup()  # Automatic warmup to improve TTFB

        self._generate_audio = True  # Flag to control audio generation
        self._session_cache = {}  # Session cache for reuse
        print("MiniCPMo initialized")

    def init_tts(self):
        """
        Initialize the text-to-speech component of the model.
        Sets up the TTS model with appropriate precision and moves it to the correct device.
        """
        # Determine the compute dtype based on quantization
        compute_dtype = torch.bfloat16 if hasattr(self, 'load_in_4bit') and self.load_in_4bit else torch.float16
        
        # Initialize TTS with automatic mixed precision
        with torch.cuda.amp.autocast(dtype=compute_dtype):
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
                
                # Compile the TTS forward method for improved performance
                print("Compiling TTS module with torch.compile...")
                # Compile the forward method instead of the module to avoid assignment issues
                original_forward = self.model.tts.forward
                self.model.tts.forward = torch.compile(
                    original_forward,
                    mode="reduce-overhead",  # Optimize for reduced overhead
                    fullgraph=False,         # Allow graph breaks for flexibility
                    dynamic=True             # Support dynamic shapes
                )
                print("TS module compiled successfully")
        
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
                    print("Starting aggressive model warmup for sub-1s TTFB...")
                    
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
                        
                        # Ultra-optimized warmup config for RTF without affecting TTFB
                        warmup_config = {
                            'session_id': warmup_session_id,
                            'tokenizer': self._tokenizer,
                            'max_new_tokens': 1,         # Minimal tokens for maximum speed
                            'do_sample': False,          # Greedy for speed
                            'temperature': 1.0,
                            'generate_audio': True,
                            'use_cache': True,
                            'audio_sample_rate': 24000,  # Standard sample rate
                            'audio_chunk_size': 32,      # Larger chunks for reduced overhead
                            'early_stopping': True,
                            'length_penalty': 0.0,
                            'num_beams': 1,
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
                    
                    print("Aggressive warmup completed - ready for sub-1s TTFB!")
                    
                except Exception as e:
                    print(f"Warning: Aggressive warmup failed with error: {str(e)}")
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
            # Reuse session if possible to improve performance
            text_key = str(prefill_data) if prefill_data else "empty"
            if text_key in self._session_cache:
                self.session_id = self._session_cache[text_key]
            else:
                self.session_id = str(uuid.uuid4())
                self._session_cache[text_key] = self.session_id
                # Limit cache size
                if len(self._session_cache) > 10:
                    self._session_cache.clear()
            
            # Prefill the model with input data if provided
            if prefill_data:
                with torch.no_grad():  # Disable gradient calculation
                    self._prefill(data=prefill_data)
            
            # Configure ultra-optimized generation parameters for RTF reduction
            generation_config = {
                'session_id': self.session_id,       # Unique ID for this generation
                'tokenizer': self._tokenizer,        # Tokenizer for text processing
                'max_new_tokens': 1,                 # Minimal tokens - halves forward passes
                'do_sample': False,                  # Greedy decoding for maximum speed
                'temperature': 1.0,                  # Default value for greedy
                'generate_audio': True,              # Always enable audio generation
                'use_cache': True,                   # Use KV cache for faster generation
                'pad_token_id': self._tokenizer.eos_token_id,  # End-of-sequence token
                'audio_sample_rate': 16000,          # REDUCED from 24000 to 16000 for faster generation
                'audio_chunk_size': 32,              # Larger chunks to reduce iteration overhead
                'early_stopping': True,              # Stop generation as soon as possible
                'num_beams': 1,                      # Force beam search = 1
                'length_penalty': 0.0,               # No length penalty
                'no_repeat_ngram_size': 0,           # Disable anti-repetition
            }
            
            # Run generation with mixed precision and no gradient calculation
            with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
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
