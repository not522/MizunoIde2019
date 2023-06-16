#!/bin/bash

current=2010-01-01
end=2011-01-01

if [ -e date.txt ]; then
    rm date.txt
fi

while [ ! $current = $end ] ; do
    echo `date -d "$current" "+%y%m%d"` >> date.txt
    current=`date -d "$current 1day" "+%Y-%m-%d"`
done

cat date.txt | parallel -j 2 -a - python3 src/main.py --format example.in --quiet --date
