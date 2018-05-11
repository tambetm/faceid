#!/bin/bash

imgdir=$1
dbpath=`realpath $2`
dbdir=`dirname $dbpath`
dbfile=`basename $dbpath`
shift 2

nvidia-docker run -t -i -v $imgdir:/mnt/images -v $dbdir:/mnt/profiles tambetm/memopol /mnt/images /mnt/profiles/$dbfile $*

