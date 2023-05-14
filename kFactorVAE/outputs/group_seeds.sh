#!/bin/bash

# assistance from ChatGPT

for subdir in ./*; do
    if [ -d "$subdir" ]; then
        base=$(basename "$subdir")
        group_dir="${base%_seed*}"
        seed="${base#*_seed}"

        # echo "Base: $base, Name: $name, Seed: $seed"
	if [[ $seed == +([0-9]) ]]; then
	    # Create or move the subdirectory into the appropriate group directory
	    # with the name "seed1", "seed2", ..., or "seed5"
	    mkdir -p "$group_dir"
	    mv "$subdir" "$group_dir/seed${seed}"
	    echo "Moved $subdir to $group_dir/seed${seed}"
	fi
    fi
done

