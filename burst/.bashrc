# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

cd /scratch/yl11330/marc/

get_burst_2gpu(){
    srun --account=yl11330 --partition c24m170-a100-2 --pty --cpus-per-task 8 --mem=32GB --time=4:00:00 --gres=gpu:2 --ntasks-per-node=1 /bin/bash
}

get_burst_4gpu(){
    srun --account=yl11330 --partition a100-4-spot    --pty --cpus-per-task 8 --mem=32GB --time=4:00:00 --gres=gpu:4 --ntasks-per-node=1 /bin/bash
}

get_sin(){
    singularity exec --nv \
    --overlay /scratch/yl11330/my_env/overlay-50G-10M-pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
}

get_conda5(){
    source /ext3/env.sh
    cd /scratch/yl11330/marc
    conda activate ./penv
}

function displaynode() {
    scontrol show node $1
}

function avail() {
    sinfo -N -p stake_a100_1,stake_a100_2,a100_1,a100_2,cilvr_a100,cilvr_a100_1,cds_a100_2,cpu_a100_1,cpu_a100_2,stake_h100_1,h100_1 -O 'NodeList:8,Partition:15,StateCompact:24,CPUsState:16,Memory:16,FreeMem:16,GresUsed:24' | grep -v '4(IDX:0-3)' | grep -E 'gpu:(a100|h100)'
}

function avail_burst() {
    sinfo -N -p c24m170-a100-2,a100-4-spot -O 'NodeList:8,Partition:15,StateCompact:24,CPUsState:16,Memory:16,FreeMem:16,GresUsed:24'
}

function gitdiff() {
    git --no-pager diff $1
}

alias displaygreenepartitions='sinfo -h -o "%P"'
alias watchq="watch -n 1 'squeue --format=\"%.8i %.5P %.80j %.5T %.7M %10R\" --me -S V'"
alias killpy="ps aux | grep python | grep -v 'grep python' | awk '{print $2}' | xargs kill -9"
alias cleanconda="conda clean --all -y"
alias wandbsync="wandb sync --clean"


MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)