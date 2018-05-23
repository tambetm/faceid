#!/bin/bash

nvidia-docker run -t -i -v /storage:/storage -p 9014:9014 tambetm/faceid $*
