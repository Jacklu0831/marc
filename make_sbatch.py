import argparse
import os
import glob

template = """#!/bin/bash

$REQUEUE
$ACCOUNT
$PARTITION
$CONSTRAINT
#SBATCH --nodes=$NODE
#SBATCH --ntasks-per-node=$NGPU
#SBATCH --cpus-per-task=$NCPU
#SBATCH --time=$TIME:00:00
#SBATCH --mem=$MEMGB
$GPULINE
#SBATCH --job-name=$JOBNAME
#SBATCH --output=slurm_outs/%j.out

module purge

MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

singularity exec --nv \\
    --overlay /scratch/zy3101/my_env/overlay-50G-10M-pytorch.ext3:ro \\
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \\
    /bin/bash -c "source /ext3/env.sh; cd /scratch/zy3101/marc; conda activate ./penv; export MASTER_PORT; \\
        $CMD"
"""

parser = argparse.ArgumentParser()
parser.add_argument('--bash_files', type=str, nargs='+', help='bash file of commands', required=True)
parser.add_argument('--gb', type=int, help='bash file of commands', default=32)
parser.add_argument('--ngpu', type=int, help='bash file of commands', default=1)
parser.add_argument('--ncpu', type=int, help='bash file of commands', default=8)
parser.add_argument('--time', type=str, help='bash file of commands', required=True)
<<<<<<< HEAD
parser.add_argument('--sbatch_dir', type=str, help='bash file of commands', default='/scratch/zy3101/marc/sbatch_files')
=======
parser.add_argument('--sbatch_dir', type=str, help='bash file of commands', default='/scratch/yl11330/marc/sbatch_files')
parser.add_argument('--burst', action='store_true')
parser.add_argument('--multi_node', action='store_true')

>>>>>>> origin/main
args = parser.parse_args()

# remove existing sbatch files
for sbatch_file in glob.glob(os.path.join(args.sbatch_dir, '*')):
    os.remove(sbatch_file)

# burst stuff
if args.burst:
    template = template.replace("$REQUEUE", "#SBATCH --requeue")
    template = template.replace("$ACCOUNT", "#SBATCH --account yl11330")
    assert args.ngpu in [1, 2, 4]
    partition = {
        1: "c12m85-a100-1",
        2: "c24m170-a100-2",
        4: "a100-4-spot",
    }[args.ngpu]
    template = template.replace("$PARTITION", f"#SBATCH --partition {partition}")
    template = template.replace("$CONSTRAINT", "")
else:
    template = template.replace("$REQUEUE", "")
    template = template.replace("$ACCOUNT", "")
    template = template.replace("$PARTITION", "")
    template = template.replace("$CONSTRAINT", "#SBATCH --constraint='a100|h100'")

if args.multi_node:
    template = template.replace('$NODE', str(args.ngpu))
    template = template.replace('$NGPU', '1')
    template = template.replace('$MEM', str(args.gb))
else:
    template = template.replace('$NODE', '1')
    template = template.replace('$NGPU', str(args.ngpu))
    template = template.replace('$MEM', str(args.gb * args.ngpu))

template = template.replace('$NCPU', str(args.ncpu))
template = template.replace('$TIME', str(args.time))

# todo: support multi-gpu
gpu_line = f'#SBATCH --gres=gpu:{args.ngpu}'
template = template.replace('$GPULINE', gpu_line)

# get job clusters from bash files
model_dirs_dict = {}
print(args.bash_files)
for bash_file in args.bash_files:
    # filter lines
    print(bash_file)
    orig_lines = open(bash_file, 'r').readlines()
    print('orig_lines')
    print(orig_lines)
    orig_lines = [l.strip() for l in orig_lines if l.strip()]
    orig_lines = [l for l in orig_lines if 'Submitted batch job' not in l]
    # collapse "\\"
    lines = []
    add_to_previous_line = False
    for l in orig_lines:
        if l.startswith('#'):
            lines.append(l)
            add_to_previous_line = False
        elif l.endswith('\\'):
            l = l[:-1]
            if add_to_previous_line:
                lines[-1] += l
            else:
                lines.append(l)
            add_to_previous_line = True
        else:
            if add_to_previous_line:
                lines[-1] += l
            else:
                lines.append(l)
            add_to_previous_line = False
    # just for assertions
    job_lines = [l for l in lines if not l.startswith('#')]
    print(job_lines)
    print('lines')
    print(lines)
    assert len(job_lines) == len(set(job_lines)), 'duplicate jobs'
    if '--tag' in job_lines[0]:
        tags = [l.split()[l.split().index('--tag') + 1] for l in job_lines]
        assert len(tags) == len(set(tags)), f'duplicate tags {tags}'
    # collect clusters
    job_cluster_names = []
    job_clusters = []
    for i, l in enumerate(lines):
        if l.startswith('#'):
            name = l[1:].strip()
            job_cluster_names.append(name)
            job_clusters.append([])
        else:
            job_clusters[-1].append(l)
        i += 1
    assert len(job_cluster_names) == len(job_clusters)
    # filter out empty clusters (random comments)
    print(job_cluster_names)
    for cluster_name, job_cluster in zip(job_cluster_names, job_clusters):
        if len(job_cluster) >= 1:
            model_dirs_dict[cluster_name] = job_cluster

print(model_dirs_dict)
sbatch_paths = []
for cluster_name, job_cluster in model_dirs_dict.items():
    for job_i, cmd in enumerate(job_cluster):
        # create sbatch content
        job_name = f"{cluster_name.replace(' ', '_')}_{job_i}"
        sbatch_content = template.replace('$JOBNAME', job_name)
        sbatch_content = sbatch_content.replace('$CMD', cmd)
        # save sbatch content
        sbatch_path = os.path.join(args.sbatch_dir, f'{job_name}.sbatch')
        print(sbatch_path)
        sbatch_paths.append(sbatch_path)
        with open(sbatch_path, 'w') as f:
            f.write(sbatch_content)

if not args.burst:
    # run sbatch
    for sbatch_path in sbatch_paths:
        os.system(f'sbatch {sbatch_path}')
