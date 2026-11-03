#!/bin/bash

# This script download a jorek unit test data.
UNITTESTNAME=`echo "$1" | sed -e 's|/$||'`
if [ -z "${DAV_URL}" ]; then
    DAV_URL="http://jorek.eu/dav_nrt"
fi
if [ -z "${UNITTESTNAME}" ]; then
  printf "Failed: requires directory name!"
fi

cd  ${UNITTESTNAME}

VERSION=""
if [ -f .version ]; then
  VERSION="`cat .version`"
fi

TGZFILE="${UNITTESTNAME}${VERSION}.tgz"

if [ -f ${TGZFILE} ]; then
  rm -f ${TGZFILE}.bck
  mv ${TGZFILE} $TGZFILE}.bck
fi

echo "Downloading ${TGZFILE}"
wget -q ${DAV_URL}/${TGZFILE}
returncode=$?
if [ $returncode -ne 0 ]; then
  cat <<EOF
####################################################################
  Failed to automatically download from web site.
  Please download reference data ${TGZFILE} yourself 
  from http:/jorek.eu/dav_nrt and
  copy it into testcases/${TESTNAME} directory.
  Launch this script again to decompress the archive.
####################################################################
EOF
  exit 1
fi

echo "Uncompress ${TGZFILE}"
tar xvzf ${TGZFILE}
