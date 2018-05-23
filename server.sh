#!/bin/bash

nvidia-docker run -t -i -v /opt/memopol3/data:/opt/memopol3/data -p 9014:80 tambetm/faceid $*
