#!/bin/bash

# Delete old drivers
echo "Removing old NVIDIA drivers..."
sudo rm -rf /usr/local/cuda*
sudo rm -f /usr/bin/nvidia-smi
sudo apt-get remove --purge '^nvidia-._' -y
sudo apt-get remove --purge '^libnvidia-._' -y
sudo apt-get remove --purge '^cuda-.*' -y

# Update repositories and clean up
echo "Updating and cleaning up..."
sudo apt update -y
sudo apt autoremove -y
sudo apt autoclean -y

# Install new driver
echo "Installing nvidia-driver-535..."
echo "Y" | sudo apt install nvidia-driver-535

echo "Done!"
