#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t tiger_nnunet_v2 "$SCRIPTPATH"
