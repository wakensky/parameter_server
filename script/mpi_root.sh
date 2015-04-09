#!/bin/bash
# set -x

if [ $# -ne 3 ]; then
    echo "usage: ./self config_dir env_name mpi_programs_dir"
    exit -1;
fi

bin_dir=`dirname "$0"`
conf=${1}/mpi.conf
mpirun=${3}//mpirun

source ${conf}

sed "s/ENV_NAME/${2}/g" ${1}/${app_conf} > ${1}/${app_conf}.${2}

my_ip=`/sbin/ifconfig ${scheduler_network_interface} | grep inet | grep -v inet6 | awk '{print $2}' | sed -e 's/[a-z]*:/''/'`
if [ -z ${my_ip} ]; then
    echo "failed to get the ip address"
    exit -1
fi

root_node="role:SCHEDULER,hostname:'${my_ip}',port:${scheduler_network_port},id:'H'"
np=$((${num_workers} + ${num_servers} + 1))

if [ ! -z ${hostfile} ]; then
    hf="-hostfile ${1}/${hostfile}"
fi

# mpirun ${hf} killall -q ps
# mpirun ${hf} md5sum ../bin/ps

${mpirun} ${hf} -np ${np} ${bin_dir}/mpi_node.${2}.sh ${root_node} ${1} ${2}
