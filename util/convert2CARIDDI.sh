#!/bin/bash

#
# Purpose: Produces .vtk files of the CARIDDI wall structures
# Requires some additional files, more info on https://www.jorek.eu/wiki/doku.php?id=cariddi-plotting
# Date: 2024-04-16
# Author: Nina Schwarz
# Heavily copying from convert2vtk.sh

# --- Cleanup things when the user presses Ctrl-C or the script finishes.
trap cleanup 1 2 3 6
function cleanup () {
  if [ "$1" != "0" ]; then
    echo ""
    echo "ABORTING."
    echo ""
  fi
  echo ""
  echo "Waiting for threads to finish..."
  wait
  if [ ! -z "$local_tmp_dir" ]; then
    rm -rf "$local_tmp_dir"
  fi
  echo "...done."
  exit $1
}

function usage () {
  echo ""
  echo "Convert JOREK restart files into 2D/3D VTK files for visualization."
  echo ""
  echo "Usage: `basename $0` [options] binary infile [extra-files]"
  echo ""
  echo "Options:"
  echo "  -dir <dir>                  Specify a custom target directory (see remarks below)"
  echo "  -j <nthreads>               Convert using <nthreads> threads [default: 1]"
  echo "  -only <step>,<step>         Convert only listed time steps"
  echo "  -only <step>-<step>         Convert only time steps in the given range"
  echo "  -only <step>-<dstep>-<step> Convert only time steps in the given range with given interval"
  echo "  -donly <dstep>              Equivalent to -only 0-<dstep>-99999"
  echo "  -time <time>,<time>         Selects time step roughly at <time> (default in JOREK-units)"
  echo "  -time <time>-<dtime>-<time> Selects time step within given time range with given interval"
  echo "  -dtime <dtime>              Equivalent to -time 0-<dtime>-infinity"
  echo "  -ms                         -time is given in milliseconds instead of in JOREK-units"
  echo "  -l                          Creates a file containing all selected timesteps and times,"
  echo "                              if parameter -(d)time is used (default:off)"
  echo "  -zip                        Compress the .vtk files using gzip"
  echo ""
  echo ""
  echo "  binary                      executable (CARIDDI_wall_curr)"
  echo ""
  echo "Remarks:"
  echo "  options are provided in CARIDDI_plot.nml as described here https://www.jorek.eu/wiki/doku.php?id=cariddi-plotting"
  echo ""
}

function mark_running () {
  ithread="$1"
  touch ${tmpdir[$ithread]}/ISRUNNING
}

function unmark_running () {
  ithread="$1"
  rm -f ${tmpdir[$ithread]}/ISRUNNING
}

function is_running () {
  ithread="$1"
  if [ -f ${tmpdir[$ithread]}/ISRUNNING ]; then
    echo 'yes'
  else
    echo 'no'
  fi
}

function get_available_thread () {
  while true; do
    for i in `seq $nthreads`; do
      if [ `is_running $i` == 'no' ]; then
        echo "$i"
        return
      fi
    done
    sleep 1
  done
}

# Find out if a time step was selected for conversion by the user (-only option)
function is_selected () {
  step_number=`echo $1 | sed -e 's/^[0 ]*\(.*.\)$/\1/'` # remove leading zeros
  step_ranges=`echo $selected_steps | tr ',' ' '` # split selected_steps, e.g., 1-3,5-7 -> 1-3 5-7
  for step_range in $step_ranges; do
    step_numbers=(`echo $step_range | tr '-' ' '`) # split step_range, e.g., 1-3 -> 1 3
    if [[ ${#step_numbers[*]} -eq 1 && ${step_numbers[0]} -eq $step_number ]] || \
       [[ ${#step_numbers[*]} -eq 2 && ${step_numbers[0]} -le $step_number && ${step_numbers[1]} -ge $step_number ]] || \
       [[ ${#step_numbers[*]} -eq 3 && ${step_numbers[0]} -le $step_number && $(($step_number % ${step_numbers[1]})) -eq 0 && ${step_numbers[2]} -ge $step_number ]] ; then
      echo "yes" # the step is contained in selected_steps, so answer 'yes'
      return
    fi
  done
  echo "no"
}

function do_convert () {
  file="$1"
  ithread="$2"
  
  cd ${tmpdir[$ithread]}
  
  stepnum=${file##*/} # Remove directory from filename
  if [ "$use_5digits" == "yes" ]; then
    stepnum=${stepnum:5:5}
  else
    stepnum=${stepnum:5:6}
  fi
  targetFile="$stepnum.vtk" # Target filename with same number as source
  targetFile="$targetDir/$targetFile" # Target filename with full path
  targetFile1="$stepnum.vtk"
  exist='true'
  pattern='false'
  for copyfile in $copyfiles; do
      cp $startDir/$copyfile .
  done
  filenames=$(grep "comp_name" CARIDDI_plot.nml | sed 's/!.*//' | awk -F "'" '{ for(i=2; i<NF; i+=2) print $i }')
  if [ -z "$filenames" ]; then
      filenames='CARIDDI_all'
  fi
  for f in $filenames
  do
      pattern='false'
      f=${f//,/}         # remove commas
      a=${f//\'/}        # remove single quotes
      for fi in ${targetDir}/"$f"*"$targetFile1" 
      do
          [[ ! -e "$fi" ]] && continue  # skip if glob doesn't match anything

          if [[ ${fi##*/} =~ ^${a}\.[0-9]+\.vtk$ || ${fi##*/} =~ ^${a}\.[0-9]+\.[0-9]+\.vtk$ ]]; then
              pattern='true'
              if  [ "$file" -nt "$fi" ]  \
	              &&  ( [ ! -z "$select_arguments" ] || [ `is_selected $stepnum` == "yes" ] ) ; then
                  exist='false'
              fi
          fi
      done
  done

  # Convert only new, selected restart files
  #   If -only flag is used, $select_arguments is empty and selection of steps is carried out below via 'is_selected'.
  #   If -time flag is used, selection of steps has happened before by python and every incoming step is accepted here.
  if ( [ "$exist" == 'false' ]  || [ "$pattern" == 'false' ] ); then
    rm -f jorek_restart.${RST_TYPE}
    ln -s $file jorek_restart.${RST_TYPE}

    if [ -f CARIDDI_plot.nml ]; then
	struct_path="${struct_path}/"
	sed -i "s@dir_struct\s*=\s*'[^']*'@dir_struct = '$struct_path'@" CARIDDI_plot.nml
    fi
    $binary  > ./log 2>&1
    if [ $? -ne 0 ]; then
      if [ ! -f $ERROR_STOP_FILE ]; then
        touch $ERROR_STOP_FILE
        cat ./log
        echo ""
        echo "ithread = $ithread"
        echo "file    = `basename $file`"
        echo ""
        echo "ERRORS OCCURED EXECUTING THE BINARY. SEE ABOVE."
      fi
      unmark_running $ithread
      return 1
    else
      egrep -i "warning|restart time" ./log
    fi

    for fi in *vtk; do
        name="${fi%.vtk}"
        mv $fi "$targetDir/$name.$targetFile1"
    done
    if [ "$zipfiles" == "yes" ]; then
      rm -f ${targetFile}.gz
      gzip $targetFile
    fi
    echo "$stepnum finished"
  fi
  unmark_running $ithread
}



SCRIPTDIR=`dirname $0`; SCRIPTDIR=`readlink -f $SCRIPTDIR`



# --- Process command line parameters
nthreads="1"
use_5digits="no"          # use old restart file index format with 5 digits

# First pass: detect -5digits early
for arg in "$@"; do
  if [ "$arg" == "-5digits" ]; then
    use_5digits="yes"
  fi
done

if [ "$use_5digits" == "yes" ]; then
  selected_steps="0-99999"
else
  selected_steps="0-999999"
fi
select_arguments=""
customdir=""
while [ $# -gt 1 ]; do
  if [ "$1" == "-j" ]; then
    nthreads="$2"
    shift 2
  elif [ "$1" == "-only" ]; then
    selected_steps="$2"
    shift 2
  elif [ "$1" == "-donly" ]; then
    if [ "$use_5digits" == "yes" ]; then
      selected_steps="0-$2-99999"
    else
      selected_steps="0-$2-999999"
    fi
    shift 2
  elif ( [ "$1" == "-time" ] || [ "$1" == "-dtime" ] ) ; then
    select_arguments="$select_arguments $1 $2"
    shift 2
  elif [ "$1" == "-ms" ]; then
    select_arguments="$select_arguments $1"
    shift 1
  elif [ "$1" == "-l" ]; then
    select_arguments="$select_arguments $1"
    shift 1
  elif [ "$1" == "-dir" ]; then
    customdir="$2"
    shift 2
  elif [ "$1" == "-zip" ]; then
    zipfiles="yes"
    shift 1
  elif [ "$1" == "-5digits" ]; then
    use_5digits="yes"
    shift
  elif [ "$1" == "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
  elif [ "${1:0:1}" == "-" ]; then
    echo "ERROR: Unknown option "$1"."
    usage
    exit 1
  else
    break
  fi
done


# --- Some parameter checks
if [ $# -lt 1 ]; then
  echo "ERROR: Not enough parameters."
  usage
  exit 1
fi
if [ ! -z "$select_arguments" ] && [[ "$select_arguments" != *"time"* ]]; then
  echo "WARNING: -l and -ms parameters will be ignored, if -(d)time is not set."
  select_arguments=""
fi
if [ "$use_5digits" == "yes" ]; then
  regexp_steps="^[0-9]{1,5}(-[0-9]{1,5}){0,2}(,[0-9]{1,5}(-[0-9]{1,5}){0,2})*$"
else
  regexp_steps="^[0-9]{1,6}(-[0-9]{1,6}){0,2}(,[0-9]{1,6}(-[0-9]{1,6}){0,2})*$"
fi
if [[ ! "$selected_steps" =~ $regexp_steps   ]]; then
  echo "ERROR: -(d)only-parameter given in wrong format."
  usage
  exit 1
fi


binary=`readlink -f $1`
shift
sourceDir=`readlink -f .`
copyfiles="$copyfiles $@"
for copyfile in $copyfiles; do
  if [ ! -f "$copyfile" ]; then
    echo "ERROR: Extra-file '$copyfile' does not exist."
    usage
    exit 1
  fi
done
echo "extra files" $copyfiles




# --- Determine output directory
if [ ! -z "$customdir" ]; then
  dir="$customdir"
else
    dir="./CARIDDI_plot"
    threeD="no"
    target="no"
fi



# --- Some basic checks
if [ ! -f $binary ]; then
  echo "ERROR: $binary does not exist."
  usage
  exit 1
elif [ ! -d $sourceDir ]; then
  echo "ERROR: $sourceDir does not exist."
  usage
  exit 1
elif [ ! -d $targetDir ]; then
  echo "ERROR: $targetDir does not exist."
  usage
  exit 1
fi



# ---- Detect restart file type
. ${SCRIPTDIR}/detect_rst_type.sh -d5 $use_5digits
if [ "$RST_TYPE" != "h5" ] && [ "$RST_TYPE" != "rst" ]; then
  echo "ERROR: RST_TYPE not detected properly: $RST_TYPE"
  usage
  exit 1
fi



# --- Select files for conversion
if [ -z "$select_arguments" ]; then
  file_available_restarts="available_restart_files.txt"
  if [ "$use_5digits" == "yes" ]; then
    ls -1 $sourceDir/jorek?????.${RST_TYPE} > $file_available_restarts
  else
    ls -1 $sourceDir/jorek??????.${RST_TYPE} > $file_available_restarts
  fi
else
  files=`${SCRIPTDIR}/select_restart_files.sh $select_arguments`
  if [ "${files:0:5}" == "ERROR" ] ; then
    echo "$files" | head -1
    usage
    exit 1
  fi
fi


# --- Create directory for vtk files
echo "Writing files to dir='$dir'."
startDir=`pwd`
mkdir -p $dir || exit 1
targetDir=`readlink -f $dir`



# --- Create local temporary directory
local_tmp_dir0="tmp_CAR_$$"
mkdir -p $local_tmp_dir0
local_tmp_dir=`readlink -f $local_tmp_dir0` # absolute path
ERROR_STOP_FILE="$local_tmp_dir/ERROR_STOP"



if [ -f "CARIDDI_plot.nml" ]; then
  # If parameters -nsub, -i_tor, -i_plane were not provided, but
  # a vtk.nml file exists, include it automatically
  struct_path=$(grep -oP "^\s*dir_struct\s*=\s*'\K[^']+" CARIDDI_plot.nml )
  if [ -z $struct_path ]; then
      struct_path="./"
  fi
  struct_path=$(readlink -f $struct_path)
  copyfiles="$copyfiles CARIDDI_plot.nml"
else
    vtk_nml="$local_tmp_dir0/CARIDDI_plot.nml"
    echo "&CARIDDI_plot"               > $vtk_nml
    echo "/"                        >> $vtk_nml
    struct_path='./'
    copyfiles="$copyfiles $vtk_nml"
fi



# --- Prepare thread temporary folders
for i in `seq $nthreads`; do
  tmpdir[$i]="$local_tmp_dir/thread_$i"
  mkdir -p ${tmpdir[$i]}
done

# --- Create a list of available selected files ---------------------------------
if [[ "$selected_steps" == "0-99999" || "$selected_steps" == "0-999999" ]]; then
  selected_available_files=$file_available_restarts
else
  file_selected_restarts="selected_restart_files.txt"
  rm -f $file_selected_restarts
  step_ranges=`echo $selected_steps | tr ',' ' '`
  for step_range in $step_ranges; do
    step_numbers=(`echo $step_range | tr '-' ' '`) # split step_range, e.g., 1-3 -> 1 3
    if [[ ${#step_numbers[*]} -eq 1 ]]; then
      istart=${step_numbers[0]};   istep=1;                    iend=${step_numbers[0]}
    elif [[ ${#step_numbers[*]} -eq 2 ]]; then
      istart=${step_numbers[0]};   istep=1;                    iend=${step_numbers[1]}
    elif [[ ${#step_numbers[*]} -eq 3 ]]; then
      istart=${step_numbers[0]};   istep=${step_numbers[1]};   iend=${step_numbers[2]}
    fi

    for i in `seq $istart $istep $iend`; do
      if [ "$use_5digits" == "yes" ]; then
        padnumber=`printf "%05d" $i`
      else
        padnumber=`printf "%06d" $i`
      fi
      echo $padnumber >> $file_selected_restarts
    done
  done

  selected_available_files='selected_available_files.txt'

  grep -f $file_selected_restarts $file_available_restarts > $selected_available_files
  rm -f $file_selected_restarts
fi
# ------------------------------------------------------------------------------


# --- Parallel file conversion
echo ""
while IFS= read -r file; do
  if [ -f "$ERROR_STOP_FILE" ]; then cleanup; fi
  ithread=`get_available_thread`
  if [ ! -f "$ERROR_STOP_FILE" ]; then
    mark_running $ithread
    do_convert $file $ithread  &
  fi
done < $selected_available_files
rm -f $file_available_restarts $selected_available_files



sleep 1
cleanup 0
