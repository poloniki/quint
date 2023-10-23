import os
from pydub import AudioSegment
import logging

if os.environ.get("ENV") == "JAX":
    try:
        from whisper_jax import FlaxWhisperPipline
        from jax.experimental.compilation_cache import compilation_cache as cc
        import jax.numpy as jnp

    except ImportError:
        print("Jax library not installed!")
else:
    try:
        import whisper
    except ImportError:
        print("Whisper library not installed!")


class Transcriber:
    """
    A class to handle Whisper transcription tasks.

    Attributes:
    - GPU_TYPE: GPU type, obtained from the environment.
    - pipeline: The Whisper pipeline used for transcription.

    Methods:
    - initialize_pipeline: Initialize the Whisper pipeline based on GPU type.
    - transcribe: Transcribe a given audio file.
    """

    def __init__(self, cache_dir="./jax_cache"):
        """Initialize the Transcriber class."""
        if os.environ.get("ENV") == "JAX":
            self.GPU_TYPE = os.environ.get("GPU_TYPE", "UNKNOWN")
            cc.initialize_cache(cache_dir)
            self.pipeline = self.initialize_pipeline()

            # Precompilation using silence; just for warming up.
            self._warmup()
            logging.info("Succesfully initialized Whisper Jax model")
        else:
            self.model = whisper.load_model("large-v2")
            logging.info("Succesfully initialized Whisper model")

    def initialize_pipeline(self):
        """
        Initialize the Whisper pipeline.

        This method checks the GPU type and initializes the pipeline
        with the appropriate data type (float16 or bfloat16).

        Returns:
        - pipeline: The initialized Whisper pipeline.
        """
        dtype = jnp.bfloat16 if self.GPU_TYPE == "A100" else jnp.float16
        try:
            pipeline = FlaxWhisperPipline(
                "openai/whisper-large-v2", dtype=dtype, max_length=64
            )
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Error initializing Whisper pipeline: {str(e)}")

    def _warmup(self):
        """
        Precompile pipeline using a silent audio.

        This is an internal method and shouldn't be called externally.
        """
        silence = AudioSegment.silent(duration=1000)  # 1 second of silence
        silence_path = "empty.flac"
        silence.export(silence_path, format="flac")

        try:
            self.pipeline(silence_path, task="transcribe", return_timestamps=True)
            os.remove(silence_path)  # Optionally remove the temporary silent file
        except Exception as e:
            raise RuntimeError(f"Error during precompilation/warmup: {str(e)}")

    def transcribe(self, audio_path):
        """
        Transcribe the given audio file.

        Parameters:
        - audio_path: Path to the audio file.

        Returns:
        - result: Transcription result.
        """
        try:
            if os.environ.get("ENV") == "JAX":
                return self.pipeline(
                    audio_path, task="transcribe", return_timestamps=True
                )
            else:
                return self.model.transcribe(audio_path, verbose=True)
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {str(e)}")
