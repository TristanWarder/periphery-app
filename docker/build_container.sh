docker run --net=host --privileged -v /dev/bus/usb:/dev/bus/usb -v /tmp/argus_socket:/tmp/argus_socket --runtime nvidia -d -t periphery:latest bash
