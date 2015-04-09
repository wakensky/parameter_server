#!/usr/bin/env python

from __future__ import division
import os,sys,time,datetime,optparse,subprocess,commands,zookeeper,copy,traceback
import multiprocessing
import Queue
from conf_parser import ConfParser, ConfError


def manual():
    pass

def install_worker(hosts_queue, conf_parser, options):
    while True:
        host = ''
        try:
            host = hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break   

        OPAGENT_SSH_PREFIX = 'sudo -u OPAGENT ssh -o "StrictHostKeyChecking no" '
        OPAGENT_SCP_PREFIX = 'sudo -u OPAGENT scp -o "StrictHostKeyChecking no" '

        base_dir_list = conf_parser.conf_dict['candidate_disks'].split(',')
        for base_dir in base_dir_list:
            base_dir = base_dir.strip()

            # Check base directory exists
            cmd =  OPAGENT_SSH_PREFIX + ' {} "'.format(host)
            cmd += ' test -d {}" '.format(base_dir)
            status, output = commands.getstatusoutput(cmd)
            if 0 != status:
                sys.stderr.write('basedir[{}] on host[{}] does not exist; [{}] skipped'.format(
                    base_dir, host, output))
                continue

            # Create data directories under base_dir
            #   env/training_data, env/models_dumped, env/models_to_neo
            cmd =  OPAGENT_SSH_PREFIX + ' %s "' % host
            cmd += ' sudo mkdir -p %s/%s/training_data ' % (base_dir, options.env)
            cmd += ' %s/%s/models_dumped ' % (base_dir, options.env)
            cmd += ' %s/%s/models_to_neo && ' % (base_dir, options.env)
            cmd += ' sudo chmod a+w %s/%s/training_data ' % (base_dir, options.env)
            cmd += ' %s/%s/models_dumped ' % (base_dir, options.env)
            cmd += ' %s/%s/models_to_neo ' % (base_dir, options.env)
            cmd += ' %s/%s " ' % (base_dir, options.env)
            status, output = commands.getstatusoutput(cmd)
            if 0 != status:
                sys.stderr.write('create data directories under base_dir [%s] on host [%s] failed; [%s]; skipped\n' % 
                    base_dir, host, output)
                continue

        # Create work directory
        cmd = OPAGENT_SSH_PREFIX + ' %s "' % host
        cmd += ' sudo mkdir -p %s && ' % (conf_parser.conf_dict['work_dir'])
        cmd += ' sudo chmod a+w %s " ' % (conf_parser.conf_dict['work_dir'])
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('create env [%s] directory under work_dir [%s] on host [%s] failed. [%s]; FATAL\n' % 
                (options.env, conf_parser.conf_dict['work_dir'], host, output))
            sys.exit(-1) 

        # Deploy centos_lib to remote work dir
        # Make sure library is a tar.gz file
        if not conf_parser.conf_dict['library'].endswith('centos_lib.tar.gz'):
            sys.stderr.write('library is not a centos_lib.tar.gz file [%s]; FATAL' % conf_parser.library)
            sys.exit(-1)
        # Copy centos_lib.tar.gz to remote
        cmd = OPAGENT_SCP_PREFIX + ' %s %s:%s/' % (
            conf_parser.conf_dict['library'], host, 
            conf_parser.conf_dict['work_dir'])
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('scp library failed [%s] [%s] FATAL\n' % 
                (conf_parser.conf_dict['library'], output))
            sys.exit(-1)
        # Uncompress centos_lib.tar.gz remotely
        cmd =  OPAGENT_SSH_PREFIX + ' %s "' % (host)
        cmd += ' tar -xzf %s//centos_lib.tar.gz -C %s/ && ' % (
            conf_parser.conf_dict['work_dir'], 
            conf_parser.conf_dict['work_dir'], )
        cmd += ' chmod a+w %s//centos_lib " ' % (
            conf_parser.conf_dict['work_dir'])
        status, output  = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('uncompress centos_lib.tar.gz failed on host [%s] [%s]; FATAL' % (
                host, output))
            sys.exit(-1)
        sys.stdout.write('Finished on host [{}]\n'.format(host))

def neo_install_worker(neo_hosts_queue, conf_parser, options):
    while True:
        host = ''
        try:
            host = neo_hosts_queue.get_nowait()
        except Queue.Empty:
            break
        if not host:
            break

        OPAGENT_SSH_PREFIX = 'sudo -u OPAGENT ssh -o "StrictHostKeyChecking no" '
        OPAGENT_SCP_PREFIX = 'sudo -u OPAGENT scp -o "StrictHostKeyChecking no" '

        # Create model directory
        neo_model_dir = conf_parser.conf_dict['neo_model_dir']
        cmd =  OPAGENT_SSH_PREFIX + ' %s "' % host
        cmd += ' sudo mkdir -p %s && sudo chmod a+w %s"' % (neo_model_dir, neo_model_dir)
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('create neo model directory failed on host [%s] [%s]; FATAL\n' % (
                host, output))
            sys.exit(-1)
        # Create merged directory
        cmd =  OPAGENT_SSH_PREFIX + ' {} "'.format(host)
        cmd += ' sudo mkdir -p {d}/merged/ && sudo chmod a+w {d}/merged/"'.format(
            d=conf_parser.conf_dict['neo_model_dir'])
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('create merged direcroty failed on host [{}] [{}]; FATAL\n'.format(host, output))
            sys.exit(-1)

        # Make sure library is a tar.gz file
        if not conf_parser.conf_dict['library'].endswith('centos_lib.tar.gz'):
            sys.stderr.write('library is not a centos_lib.tar.gz file [%s]; FATAL' % conf_parser.library)
            sys.exit(-1)
        # Deploy centos_lib
        cmd =  OPAGENT_SCP_PREFIX + ' {local_file} {host}:{target}/'.format(
            local_file=conf_parser.conf_dict['library'], 
            host=host, 
            target=conf_parser.conf_dict['neo_model_dir'])
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('scp library failed [{}] [{}] FATAL\n'.format(
                conf_parser.conf_dict['library'], output))
            sys.exit(-1)
        # Uncompress centos_lib.tar.gz remotely
        cmd =  OPAGENT_SSH_PREFIX + ' {} "'.format(host)
        cmd += ' tar -xzf {remote_dir}/centos_lib.tar.gz -C {remote_dir}/ && '.format(
            remote_dir=conf_parser.conf_dict['neo_model_dir'])
        cmd += ' chmod a+w {}/centos_lib " '.format(conf_parser.conf_dict['neo_model_dir'])
        status, output = commands.getstatusoutput(cmd)
        if 0 != status:
            sys.stderr.write('uncompress centos_lib.tar.gz failed on host [{}]. [{}] FATAL'.format(
                host, output))
            sys.exit(-1)

        sys.stdout.write('Installed NEO model directory on [{}]\n'.format(host))


def install(conf_parser, options):
    # Installation on Morpheus cluster
    host_queue = multiprocessing.Queue()
    for host in conf_parser.hosts_list:
        host_queue.put_nowait(host)

    p_list = []
    for idx in range(options.parallel):
        p = multiprocessing.Process(target=install_worker, args=(host_queue, conf_parser, options))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    print 'installed on Morpheus cluster'

    # Installation on NEO cluster
    if conf_parser.neo_ip_list:
        neo_host_queue = multiprocessing.Queue()
        for neo_host in conf_parser.neo_ip_list:
            neo_host_queue.put_nowait(neo_host)

        p_list = []
        for idx in range(options.parallel):
            p = multiprocessing.Process(target=neo_install_worker, args=(neo_host_queue, conf_parser, options))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        print 'installed on NEO cluster'
    else:
        sys.stderr.write('WARNING: Will not install on NEO cluster\n')

    return True


if '__main__' == __name__:
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('--config-dir', action='store', dest='config_dir', default='')
    opt_parser.add_option('--env', action='store', dest='env', default='')
    opt_parser.add_option('--parallel', action='store', dest='parallel', type='int', default=4)
    options, args = opt_parser.parse_args()

    print 'Try to get OPAGENT priviledge'
    cmd = 'sudo -u OPAGENT uname'
    os.system(cmd)
    print 'Got OPAGENT priviledge'

    try:
        conf_parser = ConfParser(options.config_dir, options.env)
        if not install(conf_parser, options):
            sys.syserr.write('Installation failed\n')
            exit(-1)
    except ConfError as e:
        sys.stderr.write('Parse configuration files failed [{}]\n'.format(e))
        traceback.print_exc()
        exit(-1)
    except Exception as e:
        sys.stderr.write('General exception captured [{}]\n'.format(e.args))
        traceback.print_exc()
        exit(-1)

    sys.stdout.write('Installation Finished Successfully\n')
    exit(0)
