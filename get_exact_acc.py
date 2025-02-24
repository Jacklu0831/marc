import json
import glob
import os

job_dir = './encoder_decoder/outputs/0218_ar_base/'
job_dir = './encoder_decoder/outputs/0218_noprogram_base/'

sub_prefix = 'eval_eval'
print(job_dir)
print()

json_files = list(glob.glob(f'{os.path.join(job_dir, sub_prefix)}*pred_gt.json'))
json_files.sort(key=lambda p: int(p.split('/')[-1].split('_')[2]))
jsons = [json.load(open(f, 'r')) for f in json_files]
for f, j in zip(json_files, jsons):
    num_correct = 0
    correct_task_ids = []
    for task_id, pred_gt in j.items():
        assert len(pred_gt) == 1
        pred_gt = pred_gt[0]
        num_correct += (pred_gt[0] == pred_gt[1])
        if pred_gt[0] == pred_gt[1]:
            correct_task_ids.append(task_id)
    print(f, f'{num_correct}/{len(j)}', num_correct / len(j))
    print(sorted(correct_task_ids))
    print()

# ['070dd51e-0', '0c786b71-0', '0c9aba6e-0', '1d0a4b61-0', '31d5ba1a-0', '31d5ba1a-1', '34b99a2b-0', '4852f2fa-0', '48f8583b-0', '4aab4007-0', '5b6cbef5-0', '5d2a5c43-0', '60c09cac-0', '642d658d-0', '66f2d22f-0', '833dafe3-0', '903d1b4a-0', '981571dc-0', 'aa18de87-0', 'af22c60d-0', 'b7cb93ac-0', 'bbb1b8b6-0', 'c663677b-0', 'c7d4e6ad-0', 'ca8f78db-0', 'cd3c21df-0', 'e345f17b-0', 'e345f17b-1', 'e41c6fd3-0', 'e66aafb8-0', 'e95e3d8e-0', 'ea959feb-0']
# ['070dd51e-0', '0c786b71-0', '195ba7dc-0', '1c0d0a4b-0', '1d0a4b61-0', '332efdb3-0', '3b4c2228-0', '4aab4007-0', '506d28a5-0', '59341089-0', '60a26a3e-0', '60c09cac-0', '6df30ad6-0', '770cc55f-0', '917bccba-0', 'aa18de87-0', 'bbb1b8b6-0', 'bbb1b8b6-1', 'bc4146bd-0', 'c48954c1-0', 'c663677b-0', 'c7d4e6ad-0', 'ca8f78db-0', 'ccd554ac-0', 'da2b0fe3-0', 'da2b0fe3-1', 'e1baa8a4-0', 'e95e3d8e-0', 'ea959feb-0', 'f5aa3634-0', 'fc754716-0']