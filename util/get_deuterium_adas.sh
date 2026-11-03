#!/bin/bash

# This script downloads the deuterium ADAS data into your running directory
if [ -z "${DAV_URL}" ]; then
    DAV_URL="http://jorek.eu/dav_nrt"
fi

TGZFILE="deuterium_adas.tgz"

if [ ! -f ${TGZFILE} ]; then 
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
else 
    cat <<EOF
####################################################################
 No downloading performed because ${TGZFILE} already
 exists, remove it if you want to download:
   rm testcases/${TESTNAME}/${TGZFILE}
####################################################################
EOF
    exit 1
fi

echo "Uncompress ${TGZFILE}"
tar xvzf ${TGZFILE}
