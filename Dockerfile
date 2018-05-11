FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update --fix-missing
RUN apt-get install -y python-opencv python-pip
RUN pip install python-magic exifread
RUN apt-get install -y cmake
RUN pip install dlib --install-option '--yes' --install-option USE_AVX_INSTRUCTIONS
RUN apt-get install -y git ca-certificates
RUN git clone https://github.com/tambetm/memopol.git
COPY *.py memopol/
WORKDIR memopol

ENTRYPOINT ["python","scan.py"]
