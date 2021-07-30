FROM python:slim
RUN mkdir /app
WORKDIR /app
COPY ./src/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY ./src/bot.py /app
COPY ./src/persistent/ /app/persistent/
CMD [ "python","bot.py" ]