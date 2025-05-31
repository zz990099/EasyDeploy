#!/bin/bash
################ This script should be used on host machine (not in container) ################
sudo watch -n 1 cat /sys/kernel/debug/rknpu/load
