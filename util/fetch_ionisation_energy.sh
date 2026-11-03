#!/usr/bin/env bash
# Retrieve the ionisation energy from the NIST Atomic Spectra Database
# Takes an element symbol (like W) and makes a POST request to the NIST
# website to retrieve the data.
#

set -o pipefail
set -o errtrace
set -o nounset
set -o errexit

function usage() {
  echo "Retrieve ionisation energies from the NIST Atomic Spectra Database site"
  echo "and store them in files (ion_\$symbol.dat) in the current directory."
  echo ""
  echo "Usage: $0 symbol [symbol2] [...]"
  echo "  where symbol is an element like W"
  echo ""
  echo "Examples:"
  echo "  $0 W"
  echo "  $0 H He"
  echo ""
  echo "Options:"
  echo "  -h, --help: show this message"
}

if [ "$#" -le 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  usage && exit
fi

for element in "$@"; do
  wget --post-data="encodedlist=XXT2&spectra=$element&units=1&format=1&order=0&ion_charge_out=on&e_out=0&submit=Retrieve+Data" \
    https://physics.nist.gov/cgi-bin/ASD/ie.pl -O- |\
    grep -Pazo "(?s)(?<=<pre>).*(?=</pre>)" | sed -e 's/<[^>]*>//g' | grep -ae '^  ' | tr -d '|[]()' \
    > ion_$element.dat
  # See https://stackoverflow.com/questions/3717772/regex-grep-for-multi-line-search-needed#7167115
  # works since there is only 1 pre tag
  # select the content of the pre tag, remove any html tags inside that and delete lines which do not start with 2 spaces
  # then remove all separators and special characters
  echo "Written energy levels to ion_$element.dat"
done
