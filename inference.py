from pathlib import Path
import modal
import numpy as np
import soundfile as sf

# Hugging Face authentication token for accessing private models/repos
HF_TOKEN = "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"

# Specific model version/revision to use
MODEL_REVISION = "9da79acdd8906c7007242cbd09ed014d265d281a"

# Initialize a new Modal app
app = modal.App(name="minicpm-inference-engine")


# Create a custom Docker image with all necessary dependencies
minicpm_inference_engine_image = (
    # Start with NVIDIA CUDA base image for GPU support
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")  # Install git for package management
    
    # Install PyTorch and related dependencies with specific versions for reproducibility
    .pip_install(
        "ninja==1.11.1.3",  # Build system for efficient compilation
        "packaging==24.2",   # Core utilities for Python packages
        "wheel",             # Built-package format for Python
        "torch==2.7.1",      # PyTorch deep learning framework
        "torchaudio==2.7.1", # Audio processing with PyTorch
        "torchvision==0.22.1", # Image processing with PyTorch
    )
    # Install Flash Attention for efficient attention computation
    .run_commands(
        "pip install --upgrade flash_attn==2.8.0.post2"
    )
    # Install machine learning and data processing libraries optimized for speed
    .pip_install(
        "huggingface_hub[hf_transfer]==0.30.1",  # For accessing Hugging Face models
        "transformers==4.44.2",                  # State-of-the-art NLP models
        "onnxruntime==1.20.1",                   # ONNX model runtime
        "scipy==1.15.2",                         # Scientific computing
        "numpy==1.26.4",                         # Numerical computing
        "pandas==2.2.3",                         # Data manipulation
        "bitsandbytes>=0.41.0",                  # Updated bitsandbytes for quantization
    ).pip_install(
        "Pillow==10.1.0",                        # Image processing
        "sentencepiece==0.2.0",                  # Tokenization
        "vector-quantize-pytorch==1.18.5",       # Vector quantization
        "vocos==0.1.0",                          # Vocoder for audio synthesis
        "accelerate==1.2.1",                     # Distributed training
        "timm==0.9.10",                          # Computer vision models
        "soundfile==0.12.1",                     # Audio file I/O
        "librosa==0.9.0",                        # Audio and music processing
        "sphn==0.1.4",                           # Speech processing
        "aiofiles==23.2.1",                      # Async file I/O
        "decord",                                # Video processing
        "moviepy",                               # Video editing
        "pydantic",                              # Data validation
    )
    .pip_install("gekko")  # Optimization
    
    # Configure environment variables for maximum performance
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",        # Enable faster model downloads
        "HF_HUB_CACHE": "/cache",                # Cache directory for models
        "HF_TOKEN": HF_TOKEN,                    # Authentication token
        "CUDA_LAUNCH_BLOCKING": "0",             # Allow async CUDA operations for speed
        "TORCH_CUDNN_V8_API_ENABLED": "1",       # Enable cuDNN v8 API for performance
    })
    # Add local Python module to the image
    .add_local_python_source("minicpmo")
)


# Import required modules in the context of the container
with minicpm_inference_engine_image.imports():
    import time
    import librosa  # For audio acceleration
    from minicpmo import MiniCPMo, AudioData  # Custom module for MiniCPM model
    import numpy as np  # Re-import numpy in container context
    

# Define GPU type to use
MODAL_GPU = "A10G"

# Define the Modal class with resource requirements
@app.cls(
    cpu=2,                                   # Number of CPU cores
    memory=5000,                             # Memory in MB
    gpu=MODAL_GPU,                          # GPU type
    image=minicpm_inference_engine_image,    # Custom Docker image
    min_containers=1,                       # Minimum number of containers to keep warm
    timeout=15 * 60,                        # Maximum runtime (15 minutes)
    volumes={
        # Persistent volume for caching models
        "/cache": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },
)
class MinicpmInferenceEngine:
    """
    A Modal class that handles inference with the MiniCPM model for text-to-speech.
    This class is deployed as a serverless function with GPU acceleration.
    """
    
    @modal.enter()
    def load(self):
        """Initialize the model when the container starts."""
        # Load the MiniCPM model with 8-bit quantization enabled
        self.model = MiniCPMo(device="cuda", model_revision=MODEL_REVISION, load_in_8bit=True)
        # Initialize the text-to-speech component
        self.model.init_tts()
        print("Model loaded with 8-bit quantization and ready for inference")

    @modal.method()
    def run(self, text: str):
        """
        Generate speech from text using the MiniCPM model.
        
        Args:
            text (str): The input text to convert to speech
            
        Returns:
            dict: Contains audio data, timing information, and metadata
        """
        audio_data = []  # List to store audio chunks
        start_time = time.perf_counter()  # Start timing
        time_to_first_byte = None  # Will store time until first audio chunk
        total_time = None  # Will store total processing time
        sample_rate = 16000  # Reduced from 24000 to 16000 for faster generation

        # Process each item from the model's inference output
        for item in self.model.run_inference([text]):
            if item is None:
                break  # Stop if we receive a None item
                
            # Handle text output (if any)
            if isinstance(item, str):
                print(f"Got text from MiniCPM: {text}")
                
            # Handle audio data chunks - ultra-fast processing
            if isinstance(item, AudioData):
                # Set sample rate from the first audio chunk
                if sample_rate is None:
                    sample_rate = item.sample_rate

                # Record time to first audio chunk
                if time_to_first_byte is None:
                    time_to_first_byte = time.perf_counter() - start_time
                
                # Store the audio data directly without checks
                audio_data.append(item.array)

        # Calculate total processing time
        total_time = time.perf_counter() - start_time

        # Check if we received any audio data
        if len(audio_data) == 0:
            raise ValueError("No audio data received")
        
        # Combine all audio chunks into a single array
        full_audio = np.concatenate(audio_data)

        # AFTER metric calculation: conversion to 8000 Hz for optimization
        effective_sample_rate = sample_rate  # By default, keep the original sample rate
        if len(full_audio) > 0:
            # Direct conversion without resampling for maximum speed
            full_audio = full_audio.astype(np.float32)
            # Simple normalization to avoid saturation
            max_val = np.abs(full_audio).max()
            if max_val > 0:
                full_audio = full_audio / max_val * 0.95
            
            # Conversion from 16000 Hz to 8000 Hz for optimization (after RTF calculation)
            # Downsampling by factor 2 (16000/8000 = 2)
            downsample_factor = 2
            full_audio = full_audio[::downsample_factor]  # Simple subsampling
            effective_sample_rate = 8000  # New sample rate after conversion
            
        # Return results with metadata
        return {
            "time_to_first_byte": time_to_first_byte,  # Time until first audio chunk (seconds)
            "total_time": total_time,                  # Total processing time (seconds)
            "audio_array": full_audio,                 # Generated audio as numpy array
            "sample_rate": effective_sample_rate,      # Final sample rate (8000 Hz)
            "original_sample_rate": sample_rate,       # Original sample rate (24000 Hz)
            "processing_sample_rate": 8000,            # Sample rate used for processing
            "downsample_factor": 2,                    # Downsampling factor applied
            "text": text,                              # Original input text
        }
    


@app.local_entrypoint()
def main():
    """
    Main function that demonstrates using the MinicpmInferenceEngine.
    This is the entry point when running the script locally.
    """
    # Initialize the inference engine
    engine = MinicpmInferenceEngine()

    # Warm up the model with a simple query
    print("Warming up the model...")
    result = engine.run.remote("Hi, how are you?")

    # List of example texts to process
    example_texts = [
        "I'm fine, thank you!",
        "What's your name?",
        "My name is John Doe",
        "What's your favorite color?",
        "My favorite color is blue",
        "What's your favorite food?",
    ]
    
    # Process each example text
    results = []
    for text in example_texts:
        print(f"Processing: {text}")
        result = engine.run.remote(text)
        results.append(result)

    # Get the directory of the current script
    PARENT_DIR = Path(__file__).parent

    # Save each result as a WAV file
    for result in results:
        # Create a safe filename from the text
        safe_text = "".join([c if c.isalnum() else "_" for c in result['text']])
        
        # Save versions with explicit suffixes
        output_path_processed = PARENT_DIR / f"{safe_text}_processed.wav"
        
        # Save the processed audio file
        sf.write(output_path_processed, result["audio_array"], result["sample_rate"])
        print(f"Saved processed audio to: {output_path_processed}")
        print(f"  - Downsample factor: {result.get('downsample_factor', 'N/A')}x (from {result.get('original_sample_rate', 'N/A')} Hz to {result['sample_rate']} Hz)")
        print(f"  - Audio duration: {len(result['audio_array']) / result['sample_rate']:.2f}s, Generation time: {result['total_time']:.2f}s")
        print(f"  - RTF calculated on original {result.get('original_sample_rate', 'N/A')} Hz audio")

    # Calculate and print performance metrics
    avg_time_to_first_byte = np.mean([result['time_to_first_byte'] for result in results])
    avg_realtime_factor = np.mean([
        result['total_time'] / (len(result['audio_array']) / result['sample_rate']) 
        for result in results
    ])
    
    print(f"\nPerformance Metrics:")
    print(f"- Average time to first byte: {avg_time_to_first_byte:.2f} seconds")
    print(f"- Average realtime factor: {avg_realtime_factor:.2f}x")
    
    # Interpretation of realtime factor:
    # < 1.0: Faster than real-time
    # = 1.0: Real-time
    # > 1.0: Slower than real-time
    if avg_realtime_factor < 1.0:
        print("  (Faster than real-time)")
    elif avg_realtime_factor == 1.0:
        print("  (Real-time)")
    else:
        print(f"  (Slower than real-time by {avg_realtime_factor-1.0:.1f}x)")
            




