# Extend Web demo from Caffe

This is a simple extension of the [caffe](repo/caffe/examples/)
[web demo](https://github.com/BVLC/caffe/tree/master/examples/web_demo)
to include other classification models.

NOTE: the docker image is based on the
[bvlc/caffe:cpu](https://hub.docker.com/r/bvlc/caffe/tags/)
image and that contains the [tag 1.0](https://github.com/BVLC/caffe/blob/master/docker/cpu/Dockerfile#L32-L34) of the caffe codebase. This might
not reflect what is currently on master.

## Running this

The simplest thing to get this running:

    docker-compose build
    docker-compose -f docker-compose.yml up

Then using your favourite browser, open http://localhost:5123

You might want to reload since the web servers takes a few seconds to
start since the models are initialised at start time.
