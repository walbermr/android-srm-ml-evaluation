#!/bin/bash
SOURCE_PATH=${1:-"./"}
MODE=${2:-""}

if [ "$SOURCE_PATH" == "fresh" ]
then
    MODE=${SOURCE_PATH}
    SOURCE_PATH="./"
fi

SOURCE_PATH=$(realpath ${SOURCE_PATH})

if [ "$MODE" == "fresh" -o "$(docker images -q linux:general 2> /dev/null)" == "" ]
then
    echo "Cloning and building container."
    git pull
    docker build --no-cache . -t linux:general
fi

docker run -it --rm \
    -w /root/ \
    -v "${SOURCE_PATH}:/root/" \
    --name general-ml linux:general 


DOCKER_ID=$(docker ps -a | grep general-ml | awk '{print $1}')

if [ ${DOCKER_ID} ]
then
    echo "Stopping container."
    {
        docker stop "${DOCKER_ID}"
        docker rm "${DOCKER_ID}"
    } &> /dev/null
fi

# maybe there is a better solution to that
echo "If asked, please provide your password to return files onwnership"
sudo chown -R $USER:$USER $SOURCE_PATH
