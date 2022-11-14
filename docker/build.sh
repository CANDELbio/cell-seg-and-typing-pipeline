set -ex
# SET THE FOLLOWING VARIABLES
USERNAME=gcr.io/pici-internal
# image name
IMAGE=tiff-tools
VERSION=`cat VERSION`
docker build -t $USERNAME/$IMAGE:latest -t $USERNAME/$IMAGE:$VERSION .
