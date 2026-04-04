# BBH ICL baseline for epoch sweep comparison
# makesbatch --time 10 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0328_1_epoch_sweep/bbh_icl.sh

# bbh_epochsweep_icl
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_epochsweep_icl --seed 42

# ran locally