python run.py \
    --backend gpt-3.5-turbo \
    --task_start_index 1 \
    --task_end_index 2 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 7 \
    --log logs/new_run.log \
    ${@}


python run.py --backend gpt-3.5-turbo-0613 --task_start_index 20 --task_end_index 25 --n_generate_sample 5 --n_evaluate_sample 1 --prompt_sample cot --temperature 1.0 --iterations 7 --log logs/new_run.log --algorithm tot

python run.py --backend gpt-3.5-turbo-0613 --task_start_index 0 --task_end_index 10 --n_generate_sample 5 --n_evaluate_sample 1 --prompt_sample cot --temperature 1.0 --iterations 50 --log logs/new_run.log --algorithm lats