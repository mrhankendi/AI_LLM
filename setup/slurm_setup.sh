# 1) Move private key to head node from local. Then ssh to compute node by ssh -i <private_key> <user>@<compute_node_ip>

# Set up hostnames on both nodes
# On node01 and node02 (head node and compute node) add IPs and hostnames to /etc/hosts:
127.0.0.1 localhost
10.140.83.137 node01
10.140.83.138 node02

## Passwordless SSH Setup
# Step 1: On the head node, print the public key:
cat ~/.ssh/id_rsa.pub # Copy the entire output.

# Step 2: On the compute node (node02, access it via using private key on the head node):
mkdir -p ~/.ssh
nano ~/.ssh/authorized_keys # Paste the public key at the end of that file. Save and exit.


# Then fix permissions on compute node :
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# Setup Slurm
sudo apt update
sudo apt install -y slurm-wlm munge build-essential hwloc

# Setup Munge keys

# On head node
sudo dd if=/dev/urandom bs=1 count=1024 | sudo tee /etc/munge/munge.key > /dev/null
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key
munge -n | unmunge # Test Munge
# Copy the munge key to the compute node
sudo cp /etc/munge/munge.key /home/cc/ # Copy 
sudo chown cc:cc /home/cc/munge.key
scp /home/cc/munge.key cc@node02:/tmp/

# Step 1: Create slurm.conf on Head Node
# Setup Slurm configuration
sudo mkdir -p /etc/slurm-llnl
## Example Slurm configuration file
sudo nano /etc/slurm-llnl/slurm.conf
# Paste the following configuration into slurm.conf:
# Check lscpu to set correct memory and sockets and threads
# free -m to check memory usage

# ------------------------------------------------------------------------- 
# ClusterName=chameleon-gpu
# ControlMachine=node01
# SlurmUser=slurm
# SlurmctldPort=6817
# SlurmdPort=6818
# AuthType=auth/munge
# StateSaveLocation=/var/spool/slurm-llnl
# SlurmdSpoolDir=/var/spool/slurmd
# SwitchType=switch/none
# MpiDefault=none
# SlurmctldPidFile=/var/run/slurmctld.pid
# SlurmdPidFile=/var/run/slurmd.pid
# ProctrackType=proctrack/linuxproc
# ReturnToService=1
# SchedulerType=sched/backfill
# SelectType=select/cons_res
# SelectTypeParameters=CR_Core_Memory
# GresTypes=gpu

# NodeName=node01 NodeAddr=10.140.82.148 RealMemory=128346 Sockets=2 CoresPerSocket=20 ThreadsPerCore=2 Gres=gpu:4
# NodeName=node02 NodeAddr=10.140.83.138 RealMemory=128346 Sockets=2 CoresPerSocket=20 ThreadsPerCore=2 Gres=gpu:4

# PartitionName=gpu-cluster Nodes=node01,node02 Default=YES MaxTime=INFINITE State=UP
# ------------------------------------------------------------------------- 

# Step 2: Copy slurm.conf to Compute Node
# On head nodem, copy Slurm config to compute node
scp /etc/slurm-llnl/slurm.conf cc@node02:/tmp/

# On compute node, move the Slurm config to the correct location
sudo mkdir -p /etc/slurm-llnl
sudo mv /tmp/slurm.conf /etc/slurm-llnl/

# Step 3: Create Required Directories on Both Nodes
sudo mkdir -p /var/spool/slurm-llnl /var/spool/slurmd /var/log/slurm
sudo chown slurm: /var/spool/slurm-llnl /var/spool/slurmd /var/log/slurm

# Step 4: Set Up GRES (GPU) on Both Nodes (make sure there is no extra line at the end of the file)
echo "Name=gpu Type=V100 File=/dev/nvidia0" | sudo tee /etc/slurm-llnl/gres.conf
# Sample gres.conf
# Name=gpu Type=V100 File=/dev/nvidia0
# Name=gpu Type=V100 File=/dev/nvidia1
# Name=gpu Type=V100 File=/dev/nvidia2
# Name=gpu Type=V100 File=/dev/nvidia3

# Step 5: Start Slurm Services
# On head node
sudo systemctl enable --now slurmctld
# On compute node
sudo systemctl enable --now slurmd

# Security changes
# Allow port 6817 for TCP
sudo firewall-cmd --permanent --add-port=6817/tcp

# Reload the firewall to apply changes
sudo firewall-cmd --reload

# Confirm that the port is open
sudo firewall-cmd --list-ports





