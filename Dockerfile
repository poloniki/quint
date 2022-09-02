FROM tensorflow/tensorflow:latest
COPY quint /quint
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn quint.api.fast:app --host 0.0.0.0 --port $PORT
