#!/bin/bash

CONTAINER_NAME="easy_deploy"
docker start $CONTAINER_NAME
docker exec -it $CONTAINER_NAME /bin/bash