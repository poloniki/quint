FROM tensorflow/tensorflow:latest
COPY quint /quint
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en_core_web_sm
CMD uvicorn quint.api.fast:app --host 0.0.0.0 --port $PORT


#docker run -e PORT=8000 -p 8000:8000 --env-file .env $GCR_MULTI_REGION/$PROJECT/$IMAGE
