#!/bin/bash
# set -x
if [ $# -ne 3 ]; then
    echo "usage: ./self scheduler_node config_dir env_name"
    exit -1;
fi

bin_dir=`dirname ${0}`
# support mpich and openmpi
# try mpirun -n 1 env to get all available environment
if [ ! -z ${PMI_RANK} ]; then
    my_rank=${PMI_RANK}
elif [ ! -z ${OMPI_COMM_WORLD_RANK} ]; then
    my_rank=${OMPI_COMM_WORLD_RANK}
else
    echo "failed to get my rank id"
    exit -1
fi

if [ ! -z ${PMI_SIZE} ]; then
    rank_size=${PMI_SIZE}
elif [ ! -z ${OMPI_COMM_WORLD_SIZE} ]; then
    rank_size=${OMPI_COMM_WORLD_SIZE}
else
    echo "failed to get the rank size"
    exit -1
fi

source ${2}/mpi.conf

if (( ${rank_size} < ${num_workers} + ${num_servers} + 1 )); then
    echo "too small rank size ${rank_size}"
    exit -1
fi

PRELIMINARY=""
#if [ 1 == ${my_rank} ]; then
#    PRELIMINARY=${PRELIMINARY}" HEAPPROFILE=/tmp/ps_cdn.hprof CPUPROFILE=/tmp/ps_cdn.cprof "
#    # PRELIMINARY=${PRELIMINARY}" TCMALLOC_PAGE_FENCE=1"
#fi
#if [ 173 == ${my_rank} ]; then
#    PRELIMINARY=${PRELIMINARY}" HEAPPROFILE=/tmp/ps_cdn.hprof CPUPROFILE=/tmp/ps_cdn.cprof "
#    # PRELIMINARY=${PRELIMINARY}" TCMALLOC_PAGE_FENCE=1"
#fi
#PRELIMINARY=""
#PRELIMINARY=${PRELIMINARY}" CPUPROFILE=/tmp/ps_cdn.cprof "

ulimit -c unlimited

groupname=tiger/parameter_server
if [[ -w /sys/fs/cgroup/memory/$groupname ]]; then
    echo $$ > /sys/fs/cgroup/memory/$groupname/tasks
fi

env ${PRELIMINARY} ./ps_cdn.${3} \
    -num_servers ${num_servers} \
    -num_workers ${num_workers} \
    -num_threads ${num_threads} \
    -scheduler "${1}" \
    -my_rank ${my_rank} \
    -app ${2}/${app_conf}.${3} \
    -report_interval ${report_interval} \
    ${verbose} \
    ${log_to_file} \
    ${log_instant} \
    -line_limit ${line_limit} \
    ${print_van} \
    ${shuffle_fea_id} \
    -load_data_max_mb_per_thread ${load_data_max_mb_per_thread} \
    ${key_cache} \
    ${add_beta_feature} \
    -in_memory_unit_limit ${in_memory_unit_limit} \
    -num_downloading_threads ${num_downloading_threads} \
    -keep_invalid_features_in_model ${keep_invalid_features_in_model} \
    || { echo "rank:${my_rank} launch failed"; exit -1; }
