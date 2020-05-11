FROM continuumio/anaconda3:latest
COPY . /home
RUN pip install music21
RUN pip install tornado
EXPOSE 3000