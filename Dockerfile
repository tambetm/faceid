FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y python3-pip python-opencv cmake
RUN pip3 install opencv-contrib-python python-magic exifread
RUN pip3 install dlib --install-option '--yes' --install-option USE_AVX_INSTRUCTIONS
RUN mkdir memopol
COPY *.dat memopol/
COPY *.py memopol/
WORKDIR memopol

ENTRYPOINT ["python3","scan_batch.py"]
