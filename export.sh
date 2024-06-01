#!/usr/bin/env bash

./build.sh

# docker save tiger_nnunet_v1 | gzip -c > tiger_nnunet_v1.tar.xz
docker save tiger_nnunet_v2 | gzip -c > tiger_nnunet_v2.tar.gz
