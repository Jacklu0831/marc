from collections import defaultdict
import matplotlib.pyplot as plt


slurm_path = 'slurm_outs/54877958.out' # prefixstep1
out_path = 'prefixstep1.jpg'

# slurm_path = 'slurm_outs/54877960.out' # prefixstep5
# out_path = 'prefixstep5.jpg'

# slurm_path = 'slurm_outs/54877957.out' # prefixstep10
# out_path = 'prefixstep10.jpg'

# slurm_path = 'slurm_outs/54877959.out' # prefixstep50
# out_path = 'prefixstep50.jpg'

task_ids, losses = [], []
for l in open(slurm_path, 'r').readlines():
    if not l.strip():
        continue
    # task ids
    if l.startswith('Training task '):
        task_ids.append(l.split()[-1])
    # losses
    if l.startswith("{'train_runtime':"):
        l = l[l.find('train_loss'):]
        losses.append(float(l.split()[1][:-1]))

losses = losses[:len(losses) // 2 * 2] # get rid of if only stage one is done for the last task
task_ids = task_ids[:len(losses) // 2] # get rid of last task if not done training
assert len(task_ids) * 2 == len(losses)
stage_one_losses = losses[0::2]
stage_two_losses = losses[1::2]
task_id_to_stage_one_loss = defaultdict(list)
task_id_to_stage_two_loss = defaultdict(list)
for stage_one_loss, stage_two_loss, task_id in zip(stage_one_losses, stage_two_losses, task_ids):
    task_id_to_stage_one_loss[task_id].append(stage_one_loss)
    task_id_to_stage_two_loss[task_id].append(stage_two_loss)

# get rid of partial epoch data
n_epoch = min(len(v) for v in task_id_to_stage_two_loss.values())
print('found', n_epoch, 'outer epochs')
stage_one_loss_by_epochs = []
stage_two_loss_by_epochs = []
for epoch in range(n_epoch):
    stage_one_loss_by_epoch = sum(v[epoch] for v in task_id_to_stage_one_loss.values()) / len(task_id_to_stage_one_loss)
    stage_two_loss_by_epoch = sum(v[epoch] for v in task_id_to_stage_two_loss.values()) / len(task_id_to_stage_two_loss)
    stage_one_loss_by_epochs.append(stage_one_loss_by_epoch)
    stage_two_loss_by_epochs.append(stage_two_loss_by_epoch)

plt.figure()
plt.scatter(range(n_epoch), stage_one_loss_by_epochs)
plt.scatter(range(n_epoch), stage_two_loss_by_epochs)
plt.savefig(out_path)
plt.close()