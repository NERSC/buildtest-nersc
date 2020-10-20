#!/bin/bash


JOB_ID=$(sbatch -q xfer --wrap /bin/true | awk '{print $4}')

echo JOB_ID is $JOB_ID

IS_DONE=$(sacct -j "$JOB_ID" -X -o state -p | grep -c "COMPLETED")

while [ "$IS_DONE" != 1 ]; do
    sleep 1
    IS_DONE=$(sacct -j "$JOB_ID" -X -o state -p | grep -c "COMPLETED")
done

echo checking sacct -j "$JOB_ID" -o Admincomment -n
HAS_ADMIN=$(sacct -j "$JOB_ID" -o Admincomment -n | grep -v -e "^[[:space:]]*$" | grep -c "")

if [[ "$HAS_ADMIN" -gt 0 ]]; then
    echo "Admin comment found"
fi

