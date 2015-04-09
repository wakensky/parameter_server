#!/usr/bin/env python

from __future__ import division
import os,sys,time,optparse,subprocess,commands,zookeeper,copy,getpass,time,traceback
import multiprocessing,Queue
from conf_parser import ConfParser,ConfError
from datetime import datetime,timedelta

def run(conf_parser, options, func_dict):
    # action series
    action_list = options.actions.strip().split(',')
    if not action_list:
        sys.stderr.write(
            '--actions cannot left blank; actions seperated by commas. FATAL\n'
            'legal actions [%s]\n' % (','.join(func_dict.keys())))
        return False

    for action in action_list:
        action = action.strip()

        # illegal action
        if action not in func_dict:
            sys.stderr.write(
                'action[%s] is illegal. FATAL\n'
                'legal actions [%s]\n' % (action, ','.join(func_dict.keys())))
            return False

        if func_dict[action](conf_parser, options):
            sys.stdout.write('[%s] action[%s] finished successfully\n' % (datetime.now(), action))
        else:
            sys.stderr.write('[%s] action[%s] failed; FATAL\n' % (datetime.now(), action))
            exit(-1)
    return True

def kill_worker(hosts_queue, conf_parser, options):
    while True:
        host = ''
        try:
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        SSH_PREFIX = 'ssh -o "StrictHostKeyChecking no" '
        cmd =  SSH_PREFIX + ' %s "' % host
        cmd += ' killall ps_cdn.%s" ' % options.env
        status, output = commands.getstatusoutput(cmd)
        if 0 != status and options.verbose:
            sys.stderr.write('kill ps_cdn.%s on host[%s] failed. [%s] skipped\n' % (
                options.env, host, output))


def kill(conf_parser, options):
    '''
        Kill ps_cdn processes on ps_cdn cluster
    '''
    sys.stdout.write('[{}] kill started...\n'.format(datetime.now()))
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.hosts_list:
        hosts_queue.put_nowait(host)

    p_list = []
    for idx in range(options.parallel):
        p = multiprocessing.Process(target=kill_worker, args=(hosts_queue, conf_parser, options))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    sys.stdout.write('[{}] kill finished successfully\n'.format(datetime.now()))
    return True

def clear_log_worker(hosts_queue, conf_parser, options):
    while True:
        host = ''
        try:
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        SSH_PREFIX = 'ssh -o "StrictHostKeyChecking no" '
        cmd =  SSH_PREFIX + ' %s "' % host
        cmd += ' rm -rf %s/log_* %s/van_* %s/core* %s/nohup.out %s/dashboard.out" ' % (
            conf_parser.conf_dict['work_dir'], conf_parser.conf_dict['work_dir'], 
            conf_parser.conf_dict['work_dir'], conf_parser.conf_dict['work_dir'], 
            conf_parser.conf_dict['work_dir'])
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('clear log failed on host[%s] [%s]; skipped' % (
                host, output))

        if options.verbose:
            sys.stdout.write('[{}] Cleared log on [{}] successfully\n'.format(datetime.now(), host))

def clear_log(conf_parser, options):
    '''
        Clear log files and temperary files under work directories on ps_cdn cluster
    '''
    sys.stdout.write('[{}] clear_log started...\n'.format(datetime.now()))
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.hosts_list:
        hosts_queue.put_nowait(host)

    p_list = []
    for idx in range(options.parallel):
        p = multiprocessing.Process(target=clear_log_worker, args=(hosts_queue, conf_parser, options))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    sys.stdout.write('[{}] clear-log finished successfully\n'.format(datetime.now()))
    return True

def clear_data_worker(hosts_queue, conf_parser, options):
    while True:
        host = ''
        try:
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        SSH_PREFIX = 'ssh -o "StrictHostKeyChecking no" '
        base_dir_list = conf_parser.conf_dict['candidate_disks'].split(',')
        for base_dir in base_dir_list:
            base_dir = base_dir.strip()
            cmd =  SSH_PREFIX + ' {} "'.format(host)
            cmd += ' find {base}/{env}/training_data/ -type f -maxdepth 1 | xargs rm -f" '.format(
                base=base_dir, env=options.env)
            status, output = commands.getstatusoutput(cmd)
            if 0 != status:
                sys.stderr.write('clear data under {base}/{env} on host [{host}] failed. [{detail}] skipped'.format(
                    base=base_dir, env=options.env, host=host, detail=output))

        if options.verbose:
            sys.stdout.write('[{}] Clear data on [{}] successfully\n'.format(datetime.now(), host))

def clear_data(conf_parser, options):
    '''
        Clear training/validation data on ps_cdn cluster 
        such as /data01/env/training_data/*
    '''
    sys.stdout.write('[{}] clear_data started...\n'.format(datetime.now()))
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.hosts_list:
        hosts_queue.put_nowait(host)

    p_list = []
    for idx in range(options.parallel):
        p = multiprocessing.Process(target=clear_data_worker, args=(hosts_queue, conf_parser, options))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    sys.stdout.write('[{}] clear-data finished successfully\n'.format(datetime.now()))
    return True

def clear_server_model_worker(hosts_queue, conf_parser, options):
    while True:
        host = ''
        try:
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        SSH_PREFIX = 'ssh -o "StrictHostKeyChecking no" '
        base_dir_list = conf_parser.conf_dict['candidate_disks'].split(',')
        for base_dir in base_dir_list:
            base_dir = base_dir.strip()
            cmd =  SSH_PREFIX + ' {} "'.format(host)
            cmd += ' find {base}/{env}/models_dumped/ {base}/{env}/models_to_neo/ -type f -maxdepth 1 | xargs rm -f"'.format(
                base=base_dir, env=options.env)
            status, output = commands.getstatusoutput(cmd)
            if 0 != status:
                sys.stderr.write('clear server model under {base}/{env} on host [{host}] failed. [{detail}] skipped'.format(
                    base=base_dir, env=options.env, host=host, detail=output))

        if options.verbose:
            sys.stdout.write('[{}] Clear server model on [{}] successfully\n'.format(datetime.now(), host))


def clear_server_model(conf_parser, options):
    '''
        Clear model files on ps_cdn cluster such as 
        /data06/env/models_dumped/* and /data07/env/models_to_neo/*
    '''
    sys.stdout.write('[{}] clear_server_model started...\n'.format(datetime.now()))
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.hosts_list:
        hosts_queue.put_nowait(host)

    p_list = []
    for idx in range(options.parallel):
        p = multiprocessing.Process(target=clear_server_model_worker, args=(hosts_queue, conf_parser, options))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    sys.stdout.write('[{}] clear_server_model finished successfully\n'.format(datetime.now()))
    return True

def clear_neo_model_worker(hosts_queue, conf_parser, options):
    while True:
        host = ''
        try:
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        SSH_PREFIX = 'ssh -o "StrictHostKeyChecking no" '
        cmd =  SSH_PREFIX + ' {} "'.format(host)
        cmd += ' find {remote_dir}/ -type f -maxdepth 1 | xargs rm -f " '.format(
            remote_dir=conf_parser.conf_dict['neo_model_dir'])
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('clear neo model [{remote_dir}] on host [{host}] failed; [{detail}] skipped'.format(
                remote_dir=conf_parser['neo_model_dir'], host=host, detail=output))

        if options.verbose:
            sys.stdout.write('[{}] Clear NEO model on [{}] successfully\n'.format(datetime.now(), host))

def clear_neo_model(conf_parser, options):
    '''
        Clear model files on NEO clulster
    '''
    sys.stdout.write('[{}] clear_neo_model started...\n'.format(datetime.now()))
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.neo_ip_list:
        hosts_queue.put_nowait(host)

    p_list = []
    for idx in range(options.parallel):
        p = multiprocessing.Process(target=clear_neo_model_worker, args=(hosts_queue, conf_parser, options))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    return True

def launch_one(root_node_info, my_rank, conf_parser, options, err_output_queue):
    '''
        Launch one instance remotely
        If something wrong happend, error msg will be put into err_output_queue
    '''
    # Decide ip
    remote_ip = conf_parser.hosts_list[my_rank % len(conf_parser.hosts_list)]


    # ulimit command
    # Open coredump
    ulimit_cmd = 'ulimit -c unlimited'

    # Change work dir
    # Current dir must be specified to the directory that contains centos_lib, 
    #   which acts as toolchain
    cd_cmd = 'cd {}'.format(conf_parser.conf_dict['work_dir'])

    # env for shell
    shell_env = ''
    # if 2 == my_rank or 171 == my_rank:
    #     env = "HEAPPROFILE=/tmp/ps_cdn.hprof CPUPROFILE=/tmp/ps_cdn.cprof"

    # Launch instance
    launch_cmd =  'env {} '.format(shell_env)
    launch_cmd += './ps_cdn.{} '.format(options.env)
    launch_cmd += '-num_servers {} '.format(conf_parser.conf_dict["num_servers"])
    launch_cmd += '-num_workers {} '.format(conf_parser.conf_dict["num_workers"])
    launch_cmd += '-num_threads {} '.format(conf_parser.conf_dict["num_threads"])
    launch_cmd += '-scheduler {} '.format(root_node_info)
    launch_cmd += '-my_rank {} '.format(my_rank)
    launch_cmd += '-app {}/{}.{} '.format(options.config_dir, conf_parser.conf_dict["app_conf"], options.env)
    launch_cmd += '-report_interval {} '.format(conf_parser.conf_dict["report_interval"])
    launch_cmd += '{} '.format(conf_parser.conf_dict["verbose"])
    launch_cmd += '{} '.format(conf_parser.conf_dict["log_to_file"])
    launch_cmd += '{} '.format(conf_parser.conf_dict["log_instant"])
    launch_cmd += '-line_limit {} '.format(conf_parser.conf_dict["line_limit"])
    launch_cmd += '{} '.format(conf_parser.conf_dict["print_van"])
    launch_cmd += '{} '.format(conf_parser.conf_dict["shuffle_fea_id"])
    launch_cmd += '-load_data_max_mb_per_thread {} '.format(conf_parser.conf_dict["load_data_max_mb_per_thread"])
    launch_cmd += '{} '.format(conf_parser.conf_dict["add_beta_feature"])
    launch_cmd += '-in_memory_unit_limit {} '.format(conf_parser.conf_dict["in_memory_unit_limit"])
    launch_cmd += '-num_downloading_threads {} '.format(conf_parser.conf_dict["num_downloading_threads"])
    launch_cmd += '-keep_invalid_features_in_model {} '.format(conf_parser.conf_dict["keep_invalid_features_in_model"])

    # Append specified date range
    if options.hdfs_prefix and options.last_date and options.num_days > 0:
        hdfs_dir_regex =  '\'{}/'.format(options.hdfs_prefix)
        hdfs_dir_regex += '{'

        # Date candidates
        current_date = datetime.strptime(options.last_date, '%Y%m%d')
        for i in range(options.num_days):
            hdfs_dir_regex += '{},'.format(current_date.strftime('%Y%m%d'))
            current_date -= timedelta(days=1)
        # Eliminate tailing comma, append tailing characters
        hdfs_dir_regex = hdfs_dir_regex[:-1] + '}*/instance/part-.*\''

        launch_cmd += '-hdfs_dir_regex {} '.format(hdfs_dir_regex)

    # Assembled command
    final_cmd = 'ssh -o "StrictHostKeyChecking no" tiger@{ip} "{cd};{ulimit};{launch} && exit"'.format(
        ip=remote_ip, cd=cd_cmd, ulimit=ulimit_cmd, launch=launch_cmd)

    # Execute
    if options.verbose:
        print final_cmd
    status, output = commands.getstatusoutput(final_cmd)
    if 0 != status:
        err_output_queue.put('rank[{}] failed. FATAL. [{}]'.format(
            my_rank, output))

    if options.verbose:
        sys.stdout.write('[{}] rank[{}] finished successfully\n'.format(datetime.now(), my_rank))

def copy_executable_worker(hosts_queue, src, des):
    '''
        Copy executable to work directory on remote hosts
    '''
    while True:
        host = ''
        try:
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        SSH_PREFIX = 'ssh -o "StrictHostKeyChecking no" '
        cmd =  SSH_PREFIX + ' {} "'.format(host)
        cmd += ' cp -f {src} {des}; chmod u+x {des}" '.format(src=src, des=des)
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('copy executable failed on host [{}]. [{}] FATAL\n'.format(
                host, output))
            sys.exit(-1)

def launch(conf_parser, options):
    sys.stdout.write('[{}] launch started...\n'.format(datetime.now()))

    # Copy executable to work directory on Morpheus cluster
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.hosts_list:
        hosts_queue.put_nowait(host)

    p_list = []
    src = '{}/ps_cdn'.format(conf_parser.conf_dict['bin_dir'])
    des = '{}/ps_cdn.{}'.format(conf_parser.conf_dict['work_dir'], options.env)
    for idx in range(options.parallel):
        p = multiprocessing.Process(
            target=copy_executable_worker, 
            args=(hosts_queue, src, des))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    sys.stdout.write('[{}] Deployed all executables\n'.format(datetime.now()))

    # Scheduler resides on the first host in host_list
    root_node_info = '\\"role:SCHEDULER,hostname:\'{}\',port:{},id:\'H\'\\"'.format(
        conf_parser.hosts_list[0], conf_parser.conf_dict['scheduler_network_port'])

    # Launch scheduler/workers/servers
    # Scheduler will detect servers/workers periodicly. 
    #   So if scheduler exits with return value other than 0, some nodes have failed
    num_nodes = int(conf_parser.conf_dict['num_workers']) + int(conf_parser.conf_dict['num_servers']) + 1
    p_list = []
    err_output_queue = multiprocessing.Queue()
    for i in range(num_nodes):
        p = multiprocessing.Process(target=launch_one, args=(root_node_info, i, conf_parser, options, err_output_queue))
        p.start()
        p_list.append(p)
    
    sys.stdout.write('[{}] Launched Morpheus Training System [{}]\n'.format(
        datetime.now(), options.env))
    # Check err_output_queue periodicly:
    #   Whether some nodes have been down?
    # Check Scheduler periodicly:
    #   Whether all tasks finished?
    while True:
        if not err_output_queue.empty():
            while True:
                err_info = ''
                try:
                    err_info = err_output_queue.get_nowait()
                except Queue.Empty:
                    break
                if not err_info:
                    break
                sys.stderr.write('[FATAL] Training system down: [{}]\n'.format(err_info))
            for p in p_list:
                p.terminate()
            # kill(conf_parser, options)
            return False

        p_list[0].join(1)
        if not p_list[0].is_alive():
            # Not any error message found in err_output_queue, 
            #   and Scheduler exists normally.
            sys.stdout.write('[{}] ==CONGRATS== Training system finished successfully :)\n'.format(datetime.now()))
            for p in p_list:
                p.terminate()
            # kill(conf_parser, options)
            return True

def merge_neo_model_worker(hosts_queue, conf_parser, options):
    while True:
        host = ''
        try: 
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        SSH_PREFIX = 'ssh -o "StrictHostKeyChecking no" '
        cmd =  SSH_PREFIX + ' {} "'.format(host)
        cmd += ' ulimit -c unlimited; cd {};'.format(conf_parser.conf_dict['neo_model_dir'])
        cmd += ' ./model_merger.{} -in_dir {} -in_file_pattern cdn_model_for_neo -out_dir {}"'.format(
            options.env, 
            conf_parser.conf_dict['neo_model_dir'], 
            conf_parser.conf_dict['neo_model_dir'] + '/merged/')
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('merge neo model failed on host [{}]. [{}] FATAL\n'.format(
                host, output))
            sys.exit(-1)

def merge_neo_model(conf_parser, options):
    '''
        Merge model files on NEO cluster
    '''
    sys.stdout.write('[{}] merge_neo_model started...\n'.format(datetime.now()))
    # Copy executable to model directory on Morpheus cluster
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.neo_ip_list:
        hosts_queue.put_nowait(host)
    p_list = []
    src = '{}/model_merger'.format(conf_parser.conf_dict['bin_dir'])
    des = '{}/model_merger.{}'.format(conf_parser.conf_dict['neo_model_dir'], options.env)
    for idx in range(options.parallel):
        p = multiprocessing.Process(
            target=copy_executable_worker, 
            args=(hosts_queue, src, des))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    # Merge remotely
    hosts_queue = multiprocessing.Queue()
    for host in conf_parser.neo_ip_list:
        hosts_queue.put_nowait(host)
    p_list = []
    for idx in range(options.parallel):
        p = multiprocessing.Process(target=merge_neo_model_worker, args=(hosts_queue, conf_parser, options))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    sys.stdout.write('[{}] Merge NEO model finished successfully\n'.format(datetime.now()))
    return True

if '__main__' == __name__:
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('--config-dir', action='store', dest='config_dir', default='')
    opt_parser.add_option('--env', action='store', dest='env', default='')
    opt_parser.add_option('--parallel', action='store', dest='parallel', type='int', default=4)
    opt_parser.add_option('--actions', action='store', dest='actions', default='')
    opt_parser.add_option('--verbose', action='store_const', dest='verbose', const=1)
    opt_parser.add_option('--last-date', action='store', dest='last_date')
    opt_parser.add_option('--num-days', action='store', dest='num_days', type='int', default=0)
    opt_parser.add_option('--hdfs-prefix', action='store', dest='hdfs_prefix')
    options, args = opt_parser.parse_args()

    if 'tiger' != getpass.getuser():
        sys.stderr.write('please execute under tiger\n')
        sys.exit(-1)

    func_dict = {
        'launch':               launch, 
        'kill':                 kill, 
        'clear-log':            clear_log, 
        'clear-data':           clear_data, 
        'clear-server-model':   clear_server_model, 
        'clear-neo-model':      clear_neo_model, 
        'merge-neo-model':      merge_neo_model}

    try:
        conf_parser = ConfParser(options.config_dir, options.env)
        if not run(conf_parser, options, func_dict):
            sys.stderr.write('run failed\n')
            exit(-1)
    except ConfError, e:
        sys.stderr.write('Parse configuration files failed [%s]\n' % e.args)
        traceback.print_exc()
        exit(-1)
    except Exception, e:
        sys.stderr.write('General exception captured [%s]\n' % e.args)
        traceback.print_exc()
        exit(-1)
