#mpirun -np 2 --hostfile hosts -display-allocation -display-map --allow-run-as-root -mca pml ob1 -mca btl ^openib \
#            -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket:PE=4 \
#            --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no -p 12340' \
#            -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO \
#            -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
#            -x HOROVOD_CYCLE_TIME=1 \
#            -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=120 \
#            -x MXNET_SAFE_ACCUMULATION=1 \
#	    --tag-output python -c 'import mxnet as mx; import horovod.mxnet as hvd; print(mx); hvd.init(); import socket; print(socket.gethostname())'


mpirun -np 2 --hostfile hosts -display-allocation --allow-run-as-root -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no -p 12340' --tag-output python -c 'import os; print(os.environ); import socket; print(socket.gethostname()); import mxnet as mx; import horovod.mxnet as hvd; print(mx); hvd.init(); print(hvd.rank())'
