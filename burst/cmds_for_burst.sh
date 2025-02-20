ssh burst
get_burst_1_gpu/get_burst_2_gpu/get_burst_4gpu

# setting up directory
mkdir /scratch/yl11330/marc
cd /scratch/yl11330/marc


# copying over
scp -rp yl11330@greene.hpc.nyu.edu:/scratch/yl11330/my_env /scratch/yl11330/
scp -rp yl11330@greene.hpc.nyu.edu:/scratch/yl11330/marc /scratch/yl11330/
scp -rp yl11330@greene.hpc.nyu.edu:/scratch/yl11330/marc/penv.tar /scratch/yl11330/marc/
tar -xf /scratch/yl11330/marc/penv.tar
rm /scratch/yl11330/marc/penv.tar
cp /scratch/yl11330/marc/burst/.bashrc /home/yl11330/.bashrc