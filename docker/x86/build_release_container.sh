docker run --restart unless-stopped --name periphery_release --net=host --privileged -v /dev/bus/usb:/dev/bus/usb -v /tmp/argus_socket:/tmp/argus_socket -d -t periphery_release:latest periphery
