#include <mpi.h>
#include "system/postoffice.h"
#include "system/app.h"
#include "util/local_machine.h"
#include "util/split.h"
#include "util/recordio.h"
#include "proto/instance.pb.h"

namespace PS {

DECLARE_string(center);
DECLARE_bool(verbose);

DEFINE_string(interface, "",
    "network interface will use; detect IP automatically if not specified");
DEFINE_int32(port, 8000, "network port");
DEFINE_bool(refresh_data, false,
    "true: reads training/validation data from HDFS, parses them into protobuf; "
    "false: in default");
DEFINE_string(training_begin_date, "", "begin date for training data; like 20140801");
DEFINE_int32(training_days, 0, "how many days should training data contains");
DEFINE_string(validation_begin_date, "",
    "begin date for validation data; like 20140801");
DEFINE_int32(validation_days, 0, "how many days should validation data contains");
DEFINE_string(utility_dir, "./", "utility directory contains text2proto and text2adfea");
DEFINE_int32(worker_refresh_limit, -1,
    "how many training/validation files a worker could refresh;"
    "-1 in default: no restrict");

int execute(const string& cmd, const bool show) {
    if (show) {
        LI << cmd;
    }
    return system(cmd.c_str());
}

int produceProtoData(const size_t worker_id, const string& i_am,
    const bool is_training) {

    const string HADOOP_CLIENT = "/opt/tiger/yarn_deploy/hadoop-2.3.0-cdh5.1.0/bin/hadoop";
    const string HDFS_DIR_PREFIX = "/ss_ml/recommend/train_data/batch/terafea/";
    const string HDFS_DIR_CATEGORY = "instance/";
    const uint32 SECONDS_PER_DAY = 3600 * 24;
    const string OUTPUT_DIR_PREFIX = "/data00/tiger/ps_cdn/";
    const string begin_date = is_training ?
        FLAGS_training_begin_date :
        FLAGS_validation_begin_date;
    const uint32 days = is_training ?
        FLAGS_training_days :
        FLAGS_validation_days;
    const string output_dir = is_training ?
        OUTPUT_DIR_PREFIX + "/training_data/" :
        OUTPUT_DIR_PREFIX + "/validation_data/";

    // make sure output_dir exists
    execute("mkdir -p " + output_dir, FLAGS_verbose);

    // retrieve a full data file list in specified date range
    std::vector<string> all_files;
    struct tm current_date;
    strptime(begin_date.c_str(), "%Y%m%d", &current_date);

    for (uint32 i = 0; i < days; ++i) {
        // the date of the i-th day
        char date_str[2048];
        strftime(date_str, sizeof(date_str), "%Y%m%d", &current_date);

        std::stringstream cmd;
        cmd << "HADOOP_USER_NAME=tiger " <<
            HADOOP_CLIENT << " fs -ls " <<
            HDFS_DIR_PREFIX << date_str << "*/" <<
            HDFS_DIR_CATEGORY << "part-*";
        if (FLAGS_verbose) {
            LI << cmd.str();
        }

        // download file list
        FILE *fp_pipe = popen(cmd.str().c_str(), "r");
        if (fp_pipe) {
            char path_str[8192];
            while (nullptr != fgets(path_str, sizeof(path_str), fp_pipe)) {
                const size_t path_len = strnlen(path_str, sizeof(path_str));
                if ('\n' == path_str[path_len - 1]) {
                    path_str[path_len - 1] = '\0';
                }
                string file_path(split(path_str, ' ').back());
                if (string::npos != file_path.find(HDFS_DIR_PREFIX)) {
                    all_files.push_back(file_path);
                }
            }

            pclose(fp_pipe);
        }
        else {
            LL << "executes [" << cmd.str() << "] failed";
        }

        // increments date day by day
        time_t tmp_time = mktime(&current_date);
        tmp_time += SECONDS_PER_DAY;
        current_date = *gmtime(&tmp_time);
    }

    // the file list I will process
    std::vector<string> my_files;
    const uint32 shard_size = std::ceil(
        all_files.size() / static_cast<float>(FLAGS_num_workers));
    for (size_t i = 0; i < all_files.size(); ++i) {
        if (i >= worker_id * shard_size && i < (worker_id + 1) * shard_size) {
            my_files.push_back(all_files[i]);
        }
    }

    // transform from terafea (text) to protobuf
    for (size_t i = 0; i < my_files.size(); ++i) {
        // how many data files a worker could refresh
        if (FLAGS_worker_refresh_limit >= 0 &&
            i >= static_cast<size_t>(FLAGS_worker_refresh_limit)) {
            break;
        }

        // process report
        LI << "[" << i_am << "] " <<
            "refreshing data[" << (is_training ? "training]" : "validation]") <<
            "in progress: " << i + 1 << "/" << my_files.size();

        string output_data_file_name = string("/adfea_protobuf_part_") +
            std::to_string(worker_id) + "_" + std::to_string(i);
        std::stringstream output_data_file_path;
        output_data_file_path << output_dir << output_data_file_name;
        std::stringstream cmd;
        cmd << "HADOOP_USER_NAME=tiger " << HADOOP_CLIENT <<
            " fs -cat " << my_files[i] << " | gunzip | " <<
            FLAGS_utility_dir << "./text2adfea | " <<
            FLAGS_utility_dir << "./text2proto -format 'adfea' > " <<
            output_data_file_path.str();
            execute(cmd.str(), FLAGS_verbose);

        if (!FLAGS_center.empty()) {
            // store meta info into center directory
            //  scheduler could read global key range from files in center directory

            // make sure center directory exists
            string center_dir = FLAGS_center + "/";
            if (is_training) {
                center_dir += "training_data/";
            }
            else {
                center_dir += "validation_data/";
            }
            execute("mkdir -p " + center_dir, FLAGS_verbose);

            // the first protobuf in file just produced contains meta data
            // copy it out to a new tiny file
            InstanceInfo instance_info;
            {
                File *data_file = File::openOrDie(output_data_file_path.str(), "r");
                RecordReader reader(data_file);
                reader.ReadProtocolMessage(&instance_info);
                reader.Close();  // also closes data_file
            }

            // make sure target directory exists
            const char *sub_dir = is_training ? "training_data/" : "validation_data/";
            execute("mkdir -p " + FLAGS_center + "/" + sub_dir, FLAGS_verbose);

            {
                File *new_meta_file = File::openOrDie(
                    FLAGS_center + "/" + sub_dir + output_data_file_name,
                    "w");
                RecordWriter protobuf_writer(new_meta_file);
                protobuf_writer.WriteProtocolMessage(instance_info);
                protobuf_writer.Close();  // also closes new_meta_file
                // NOTICE: the destructor of Recordwriter/Recordreader
                //  does not close File pointer it holds
            }
        }
    }

    return 0;
}

int refreshData(const size_t worker_id, const string& i_am) {
    // produce training data
    if (!FLAGS_training_begin_date.empty() &&
        0 != produceProtoData(worker_id, i_am, true)) {
        return -1;
    }

    // produce validation data
    if (!FLAGS_validation_begin_date.empty() &&
        0 != produceProtoData(worker_id, i_am, false)) {
        return -1;
    }

    return 0;
}

void Init() {
  int my_rank, rank_size;
  CHECK(!MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CHECK(!MPI_Comm_size(MPI_COMM_WORLD, &rank_size));

  // get my node
  int nserver = FLAGS_num_servers;
  int nclient = FLAGS_num_workers;
  if (my_rank == 0) CHECK_GT(rank_size, nserver + nclient);

  FLAGS_num_unused = rank_size - nserver - nclient - 1;

  string ip("");
  if (!FLAGS_interface.empty()) {
    ip = LocalMachine::IP(FLAGS_interface);
  }
  else {
    // enumerate all inet address
    // choose the first IP address that doesn't begin with "127."
    FILE *ifconfig_pipe = popen("/sbin/ifconfig", "r");
    if (nullptr != ifconfig_pipe) {
        char line[2048];
        const char *INET_PREFIX = "inet ";
        const char *ADDR_PREFIX = "addr:";

        // There are two common output stype of ifconfig; as follows:
        //  inet addr:10.4.160.9
        //  inet 10.4.160.9
        while (nullptr != fgets(line, sizeof(line), ifconfig_pipe)) {
            const char *start_inet = strstr(line, INET_PREFIX);
            if (nullptr != start_inet) {
                const char *start_ip = start_inet + strnlen(INET_PREFIX, 32);
                if (nullptr != strstr(start_inet, ADDR_PREFIX)) {
                    start_ip += strnlen(ADDR_PREFIX, 32);
                }

                // skip all IPs that begin with "127."
                if (start_ip == strstr(start_ip, "127.")) {
                    continue;
                }

                // a space is the tailing of IP string
                const char *start_space = strstr(start_ip, " ");
                if (nullptr != start_space) {
                    ip.assign(start_ip, start_space - start_ip);
                }
            }
        }

        pclose(ifconfig_pipe);
    }
  }
  CHECK(!ip.empty()) << "fail to get the ip from interface " << FLAGS_interface;

  string my_node = "role:";
  string id;
  if (my_rank == 0) {
    my_node += "SCHEDULER";
    id = "H";
  } else if (my_rank < nclient + 1) {
    my_node += "CLIENT";
    id = "W" + std::to_string(my_rank - 1);
  } else if (my_rank < nclient + nserver + 1) {
    my_node += "SERVER";
    id = "S" + std::to_string(my_rank - nclient - 1);
  } else {
    my_node += "UNUSED";
    id = "U" + std::to_string(my_rank - nclient - nserver - 1);
  }
  my_node += ",hostname:'" + ip + "',port:" +
             std::to_string(FLAGS_port+my_rank) + ",id:'" + id + "'";
  FLAGS_my_node = my_node;

  if (FLAGS_refresh_data &&
        string::npos != my_node.find("role:CLIENT") &&
        0 != refreshData(my_rank - 1, my_node)) {
    // TODO just print warning log now
    LL << "[" << my_node << "] didn't finish refreshData" << std::endl;
  }

  // send my node to the scheduler, and save into ../config/nodes
  if (my_rank == 0) {
    std::ofstream file(FLAGS_node_file);
    CHECK(file.good()) << " fail to write " + FLAGS_node_file;
    file << my_node << "\n";
    char node[100];
    MPI_Status stat;
    for (int i = 1; i < rank_size; ++i) {
      memset(node, 0, 100);
      CHECK(!MPI_Recv(node, 100, MPI_CHAR, i, 0, MPI_COMM_WORLD, &stat));
      file << node << "\n";
    }
  } else {
    int n = my_node.size();
    char send[n+5];
    memcpy(send, my_node.data(), n);
    CHECK(!MPI_Send(send, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD));
  }
}

} // namespace PS

int main(int argc, char *argv[]) {
  FLAGS_logtostderr = 1;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  CHECK(!MPI_Init(&argc, &argv));

  try {
    PS::Init();
    PS::Postoffice::instance().run();
  } catch (std::exception& e) {
    LL << e.what();
  }


  // LL << "ok";
  // MPI_Barrier(MPI_COMM_WORLD);
  // LL << "done";
  CHECK(!MPI_Finalize());
  // try {
  // }  catch (std::bad_alloc& ba) {
  //   std::cerr << "bad_alloc caught: " << ba.what() << '\n';
  // }

  return 0;
}
