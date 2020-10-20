#!/bin/sh

y=500
i=0

while [ $i -ne $y ]
do
  echo "Hello There!" >$i
  i=`expr $i + 1`
done

