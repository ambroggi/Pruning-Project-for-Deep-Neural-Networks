#!/bin/bash


# for thing in {1..14..2}; do
#     echo $thing &
#     sleep 1
#     echo $((thing+1)) &
#     wait
# done

# https://unix.stackexchange.com/a/216475
for thing in {0..12}; do
    python src/buildontology.py $thing &
    sleep 10
    python src/buildontology.py $(($thing+13)) &
    wait
done

