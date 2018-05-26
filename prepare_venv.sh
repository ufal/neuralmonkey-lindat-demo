#!/bin/bash

virtualenv -p python3 lsd-demo
source lsd-demo/bin/activate

pip install -r requirements.txt
