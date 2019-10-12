pkill python

export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"
export NUMSTEPS="${NUMSTEPS:-900000}"
export DTYPE="${DTYPE:-float16}"
export BS="${BS:-512}"
export ACC="${ACC:-1}"
export MODEL="${MODEL:-bert_24_1024_16}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:-20}"
export LR="${LR:-0.000625}"
export LOGINTERVAL="${LOGINTERVAL:-10}"
export CKPTDIR="${CKPTDIR:-ckpt_stage1_lamb}"
export CKPTINTERVAL="${CKPTINTERVAL:-300000000}"
export OPTIMIZER="${OPTIMIZER:-lamb}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.003125}"
export BYTEPS_PARTITION_BYTES="${BYTEPS_PARTITION_BYTES:-4096000}"
export BYTEPS_NCCL_GROUP_SIZE="${BYTEPS_NCCL_GROUP_SIZE:-16}"
export BPS_HOME="${BPS_HOME:-/root/byteps}"
export NVIDIA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export DMLC_WORKER_ID="${DMLC_WORKER_ID:-0}"
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
export DMLC_ROLE=worker
export NCCL_MIN_NRINGS="${NCCL_MIN_NRINGS:-16}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export OPTIONS="${OPTIONS:- }"

echo $NVIDIA_VISIBLE_DEVICES

python3 $BPS_HOME/launcher/launch.py \
	python3 run_pretraining.py \
            --data='/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train' \
            --data_eval='/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test' \
	    --optimizer $OPTIMIZER \
	    --warmup_ratio $WARMUP_RATIO \
            --num_steps $NUMSTEPS \
	    --ckpt_interval $CKPTINTERVAL \
	    --dtype $DTYPE \
	    --ckpt_dir $CKPTDIR \
	    --lr $LR \
	    --total_batch_size $BS \
	    --total_batch_size_eval $BS \
	    --accumulate $ACC \
	    --model $MODEL \
	    --max_seq_length $MAX_SEQ_LENGTH \
	    --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
	    --num_data_workers 4 \
	    --no_compute_acc --raw \
	    --comm_backend byteps --log_interval $LOGINTERVAL $OPTIONS
