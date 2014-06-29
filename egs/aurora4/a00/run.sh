#!/bin/bash

if [ ! -d log ]; then
  mkdir -p log
fi

curhost=`hostname`

if [ "${curhost}" == "atlas6-c01" ] || [ "${curhost}" == "atlas5-c01" ] || [ "${curhost}" == "gold-c01" ] ; then

  if [ $# -ge 1 ]; then
    echo "Submitting script [$@] to goldgpu queue."
    read -p "Press any key to start..."
    logfile=`echo $1 | sed "s:\.sh::g"`
    bsub -q goldgpu -o `pwd`/log/${logfile}.out -e `pwd`/log/${logfile}.err ./$@
  else
    echo "Invalid job specification: [$@]"
    exit 1;
  fi

else

  if [ $# -ge 1 ]; then
    echo "Running script [$@] locally..."
    read -p "Press any key to start..."
    logfile=`echo $1 | sed "s:\.sh::g"`
    ./$@ 2>&1 | tee -a log/${logfile}.log
  else
    echo "Invalid job specification: [$@]"
    exit 1;
  fi
fi

exit 0;

