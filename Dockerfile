FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
FROM python:3.10.6-buster
RUN apt-get update
COPY requirements.txt /requirements.txt
COPY setup.py setup.py
COPY quint /quint
RUN pip install --upgrade pip
RUN apt-get install -y ffmpeg
RUN pip install .
RUN pip install git+https://github.com/openai/whisper.git
RUN python -m spacy download en_core_web_lg
ENV GPU_TYPE=T40
EXPOSE 80
CMD ["gunicorn", "-b", "0.0.0.0:80", "quint.api.fast:app", "--workers", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "320"]
