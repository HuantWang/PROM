#!/bin/bash

if [ -d "logs" ]; then
    rm -rf logs
fi
if [ -d "models" ]; then
    rm -rf models
fi
if [ -d "saved_models" ]; then
    rm -rf saved_models
fi
exit 0
