FROM continuumio/anaconda3:latest
COPY . /home
RUN pip install music21
RUN pip install tornado
RUN conda install tensorflow
EXPOSE 3000