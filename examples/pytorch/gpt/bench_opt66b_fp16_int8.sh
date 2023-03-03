./bin/gpt_gemm 1 1 128 72 128 36864 50272 1 2 0
./bin/gpt_gemm 4 1 128 72 128 36864 50272 1 2 1
./bin/gpt_gemm 16 1 128 72 128 36864 50272 1 2 1
FMHA_ENABLE=ON mpirun -n 2 --allow-run-as-root python3 ../examples/pytorch/gpt/opt_summarization_bench.py --summarize --max_ite 20 --ft_model_location opt-66b/c-model --hf_model_name opt-66b --tensor_para_size=2
python ../examples/pytorch/gpt/utils/huggingface_opt_convert.py -i opt-66b -o opt-66b/c-model -i_g 1
FMHA_ENABLE=ON python3 ../examples/pytorch/gpt/opt_summarization_bench.py --summarize --max_ite 20 --ft_model_location opt-66b/c-model --hf_model_name opt-66b --int8_mode 1