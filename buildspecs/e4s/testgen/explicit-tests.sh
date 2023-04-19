#!/bin/bash

# Run command that returns output
output1=$(spack test list)
output2=$(spack find -x)

# Place every word containing a @ in a list for each command.
# Skip the first entry since this will be the compiler.
list1=($(echo "$output1" | grep -o '\S*@[^[:space:]]*' | tail -n +2))
list2=($(echo "$output2" | grep -o '\S*@[^[:space:]]*' | tail -n +2))

echo package:

# Print only the words that are in both lists, cutting off the version
for word in "${list1[@]}"; do
    if [[ " ${list2[@]} " =~ " ${word} " ]]; then
         echo "- ${word%%@*}"
    fi
done | sort -u
