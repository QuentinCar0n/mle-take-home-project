from pathlib import Path

import modal
import numpy as np
import soundfile as sf


HF_TOKEN = "<your-huggingface-token>"


MODEL_REVISION = "9da79acdd8906c7007242cbd09ed014d265d281a"


app = modal.App(name="minicpm-inference-engine")


minicpm_inference_engine_image = (
    # install MiniCPM-o dependencies
    modal.Image.from_registry(f"nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    # Install flash-attn dependencies
    .pip_install(  # required to build flash-attn
        "ninja==1.11.1.3",
        "packaging==24.2",
        "wheel",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "torchvision==0.22.1",
    )
    .run_commands(  # add flash-attn
        "pip install --upgrade flash_attn==2.8.0.post2"
    )
    # Install general AI dependencies
    .pip_install(
        "huggingface_hub[hf_transfer]==0.30.1",
        "transformers==4.44.2",
        "onnxruntime==1.20.1",
        "scipy==1.15.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
    ).pip_install(
        "Pillow==10.1.0",
        "sentencepiece==0.2.0",
        "vector-quantize-pytorch==1.18.5",
        "vocos==0.1.0",
        "accelerate==1.2.1",
        "timm==0.9.10",
        "soundfile==0.12.1",
        "librosa==0.9.0",
        "sphn==0.1.4",
        "aiofiles==23.2.1",
        "decord",
        "moviepy",
        "pydantic",
    )
    .pip_install("gekko")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/cache",
            "HF_TOKEN": HF_TOKEN,
        }
    )  # and enable it
    .add_local_python_source("minicpmo")
)


with minicpm_inference_engine_image.imports():
    import time
    from minicpmo import MiniCPMo, AudioData
    import numpy as np
    

MODAL_GPU = "A10G"

@app.cls(
    cpu=2,
    memory=5000,
    gpu=MODAL_GPU,
    image=minicpm_inference_engine_image,
    min_containers=1,
    timeout=15 * 60,
    volumes={
        "/cache": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },
)
class MinicpmInferenceEngine:
    @modal.enter()
    def load(self):
        self.model = MiniCPMo(device="cuda", model_revision=MODEL_REVISION)
        self.model.init_tts()


    @modal.method()
    def run(self, text: str):
        audio_data = []
        start_time = time.perf_counter()
        time_to_first_byte = None
        total_time = None

        for item in self.model.run_inference([text]):
            if item is None:
                break
            if isinstance(item, str):
                print(f"Got text from MiniCPM: {text}")
            if isinstance(item, AudioData):
                assert item.sample_rate == 24000

                if time_to_first_byte is None:
                    time_to_first_byte = time.perf_counter() - start_time
                    
                audio_data.append(item.array)

        total_time = time.perf_counter() - start_time

        if len(audio_data) == 0:
            raise ValueError("No audio data received")
        
        full_audio = np.concatenate(audio_data)

        return {
            "time_to_first_byte": time_to_first_byte,
            "total_time": total_time,
            "audio_array": full_audio,
            "sample_rate": 24000,
            "text": text,
        }
    


@app.local_entrypoint()
def main():
    engine = MinicpmInferenceEngine()

    # Warmup
    result = engine.run.remote("Hi, how are you?")

    results = []
    for text in ["I'm fine, thank you!", "What's your name?", "My name is John Doe", "What's your favorite color?", "My favorite color is blue", "What's your favorite food?", ]:
        result = engine.run.remote(text)
        results.append(result)


    PARENT_DIR = Path(__file__).parent

    for result in results:
        sf.write(PARENT_DIR / f"{result['text']}.wav", result["audio_array"], result["sample_rate"])
        print(f"Wrote {result['text']}.wav to {PARENT_DIR / result['text']}.wav")

    print(f"Time to first byte: {np.mean([result['time_to_first_byte'] for result in results])}")
    print(f"Realtime Factor: {np.mean([result['total_time'] / (len(result['audio_array']) / result['sample_rate']) for result in results])}")
            




