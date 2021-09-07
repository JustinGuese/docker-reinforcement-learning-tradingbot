FROM python:slim
RUN mkdir /app
WORKDIR /app
COPY ./src/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY ./src/livemodel.py /app
COPY ./src/a2cmlp_bestprofit.hf5 /app
CMD [ "python","livemodel.py" ]