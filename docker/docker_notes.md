docker build -t matsciml/habana .
docker run --name=<imageName>
docker cp <containerId>:/matsciml/lightning_logs ./training_results
