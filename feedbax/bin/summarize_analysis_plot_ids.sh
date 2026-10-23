#!/bin/bash
# e.g. run this in the directory 1-analysis

if [ $# -ne 1 ]; then
    echo "Usage: $0 <label_name>"
    exit 1
fi

label="$1"

rm -f output.txt
touch output.txt

for file in *.qmd; do
    echo "=== $(grep "^NB_ID = " "$file" | head -n1 | sed 's/.*NB_ID = //' | tr -d '"'"'")" >> output.txt
    grep -h "$label = " "$file" | sed "s/.*$label = //" | tr -d '"'"'" >> output.txt
    echo >> output.txt
done