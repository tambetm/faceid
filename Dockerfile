FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y python3-pip python-opencv cmake
RUN pip3 install opencv-contrib-python python-magic exifread
RUN pip3 install dlib --install-option '--yes' --install-option USE_AVX_INSTRUCTIONS
RUN pip3 install flask

RUN apt-get install -y apache2 libapache2-mod-wsgi-py3

RUN mkdir memopol
COPY *.dat memopol/
COPY *.py memopol/
COPY *.wsgi memopol/
WORKDIR memopol

# Manually set up the apache environment variables
ENV APACHE_RUN_USER www-data
ENV APACHE_RUN_GROUP www-data
ENV APACHE_LOG_DIR /var/log/apache2
ENV APACHE_LOCK_DIR /var/lock/apache2
ENV APACHE_PID_FILE /var/run/apache2.pid

# Enable debugging
ENV FLASK_ENV development

EXPOSE 80
COPY faceid.conf /etc/apache2/sites-available/000-default.conf

# By default, simply start apache.
CMD /usr/sbin/apache2ctl -D FOREGROUND
