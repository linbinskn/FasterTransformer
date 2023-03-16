./bin/gpt_gemm 1 1 128 112 128 57344 250880 1 4 0
./bin/gpt_gemm 4 1 128 112 128 57344 250880 1 4 1
./bin/gpt_gemm 16 1 128 112 128 57344 250880 1 4 1

./bin/gpt_gemm 1 1 128 112 128 57344 250880 1 8 0
./bin/gpt_gemm 4 1 128 112 128 57344 250880 1 8 1
./bin/gpt_gemm 16 1 128 112 128 57344 250880 1 8 1

./bin/gpt_gemm 1 1 128 112 128 57344 250880 1 2 0
./bin/gpt_gemm 4 1 128 112 128 57344 250880 1 2 1
./bin/gpt_gemm 16 1 128 112 128 57344 250880 1 2 1

FMHA_ENABLE=ON mpirun -n 8 --allow-run-as-root python ../examples/pytorch/gpt/bloom_lambada_bench.py --checkpoint-path /work/data2/bloom/c-model/8-gpu --tokenizer-path /work/data2/bloom --dataset-path ../datasets/lambada/lambada_test.jsonl --show-progress --int8_mode 0 --inference-data-type fp16 --weights-data-type fp16
FMHA_ENABLE=ON mpirun -n 4 --allow-run-as-root python ../examples/pytorch/gpt/bloom_lambada_bench.py --checkpoint-path /work/data2/bloom/c-model/4-gpu --tokenizer-path /work/data2/bloom --dataset-path ../datasets/lambada/lambada_test.jsonl --show-progress --int8_mode 1 --inference-data-type fp16 --weights-data-type fp16
# python3 ../examples/pytorch/gpt/utils/huggingface_bloom_convert.py --input-dir /work/data2/bloom --output-dir /work/data2/bloom/c-model -tp 2 -p 16 -v --data-type fp16
# FMHA_ENABLE=ON mpirun -n 2 --allow-run-as-root python ../examples/pytorch/gpt/bloom_lambada_bench.py --checkpoint-path /work/data2/bloom/c-model/2-gpu --tokenizer-path /work/data2/bloom --dataset-path ../datasets/lambada/lambada_test.jsonl --show-progress --int8_mode 3 --inference-data-type fp16 --weights-data-type fp16