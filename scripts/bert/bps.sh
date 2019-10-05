pkill python
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=1
export DMLC_ROLE=worker
export NCCL_MIN_NRINGS=8
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=120
export MXNET_SAFE_ACCUMULATION=1

export EVAL_TYPE=benchmark
python /opt/byteps/launcher/launch.py \
	python run_pretraining.py \
            --data='/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train' \
            --data_eval='/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test' \
	    --optimizer $OPTIMIZER \
            --num_steps $NUMSTEPS \
            --dtype $DTYPE \
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
	    --comm_backend byteps --log_interval $LOGINTERVAL

            #--synthetic_data --eval_use_npz \
