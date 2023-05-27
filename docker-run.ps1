param(
    [string]$SOURCE_PATH="./",
    [string]$MODE
)

if ("fresh" -eq $SOURCE_PATH)
{
    $MODE=${SOURCE_PATH}
    $SOURCE_PATH="./"
}

$SOURCE_PATH="$(Resolve-Path ${SOURCE_PATH})"

if ( "fresh" -eq $MODE -or  "$(docker images -q srm-eval:latest)" -eq "")
{
    Write-Host "Cloning and building container."
    git pull
    docker build --no-cache . -t srm-eval:latest
}

docker run -it --rm -w /root/srm -v "${SOURCE_PATH}:/root/srm" --name srm-container srm-eval:latest ./run_pipeline.sh

$DOCKER_ID=$(docker ps -a | Select-String "srm-container" | %{ $_.Line.Split(' ')[-1]; })

if( ${DOCKER_ID} )
{
    Write-Host "Stopping container."
    docker stop "${DOCKER_ID}"
    Write-Host "Removing container."
    docker rm "${DOCKER_ID}"
}
