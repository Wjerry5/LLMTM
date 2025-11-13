source /home/ma-user/Ascend/ascend-toolkit/set_env.sh
# source /home/ma-user/Ascend/nnal/asdsip/set_env.sh
source /home/ma-user/Ascend/nnal/atb/set_env.sh

PRECHECKPOINT_PATH="/home/ma-user/work/LLMDyG_Motif/Pangu/openPangu-Embedded-7B-DeepDiver"
export FLAGS_npu_jit_compile=false
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "$PRECHECKPOINT_PATH" \
    --served-model-name "pangu_auto" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --host "127.0.0.1" \
    --port 8888 \
    --max-num-seqs 256 \
    --max-model-len 30000 \
    --max-num-batched-tokens 4096 \
    --tokenizer-mode "slow" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.93 