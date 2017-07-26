#!/bin/bash

SCRIPT=$(readlink -f "$0")
echo $SCRIPT
BASEDIR=$(dirname "$SCRIPT")
echo $BASEDIR

LOCKFILE=/tmp/lock1.txt
if [ -e ${LOCKFILE} ] && kill -0 `cat ${LOCKFILE}`; then
    echo "already running"
    exit
fi

# make sure the lockfile is removed when we exit and then claim it
trap "rm -f ${LOCKFILE}; exit" INT TERM EXIT
echo $$ > ${LOCKFILE}

cd ${BASEDIR}
/home/ubuntu/anaconda2/bin/curl -X POST http://202.63.105.85/mintmesh/v1/enterprise/not_parsed_resumes -F authentication_key=107857d5d4be08e5e2dc51ef141e0924 > list2.json
/home/ubuntu/anaconda2/bin/python downloadAndProcessFiles.py


rm -f ${LOCKFILE}
