# buildtest-cori

This repository contains tests for Cori at NERSC. To get started you can either clone this repo via git

```
git clone https://github.com/buildtesters/buildtest-cori
```

Alternately you can clone via buildtest as follows:

```
buildtest repo add https://github.com/buildtesters/buildtest-cori
``` 

Next copy site configuration file ``settings.yml`` to buildtest location ``$HOME/.buildtest/settings.yml``

If you cloned tests via ``buildtest repo add`` buildtest will add all repository to search path. 

# Building Tests

**Note: All tests are written in YAML using .yml extension**
   
To build tests use ``buildtest build`` command for example we build all tests in ``system`` directory as follows

```
buildtest build -b /tmp/github.com/buildtesters/buildtest-cori/system/
```

Same test could be achieved by specifying relative path based on search paths 

```
buildtest build -b system
```

Alternately, you can cd into any directory and specify a relative file path

```
$ cd /tmp/github.com/buildtesters/buildtest-cori/
$ buildtest build -b system/cray_env.yml 
```

You can specify multiple buildspecs either files or directory via ``-b`` option

```
buildtest build -b slurm/partition.yml -b slurmutils/
```

You can exclude a buildspec via `-x` option this behaves same way as `-b` option so you can specify
a directory or filepath which could be absolute path, or relative path. This is useful when
you want to run multiple tests grouped in directory but exclude a few.

```
buildtest build -b slurm -x slurm/sinfo.yml
```

If you find no buildspecs due to discovery (``-b``) and exclusion you can run into no buildspecs found, this 
would happen if you do the inverse operation for example

```
buildtest build -b slurm -x slurm
buildtest build -b slurm/sinfo.yml -x slurm/sinfo.yml
```


# Finding all buildspecs

To find and validate all buildspecs run ``buildtest buildspec find`` and this will show a list of available buildspecs that
buildtest found and are validated with each schema. 

```
$ buildtest buildspec find



Detected 1 invalid buildspecs
Writing invalid buildspecs to file: /global/u1/s/siddiq90/buildtest/buildspec.error 



Name                      Type                      Buildspec                
________________________________________________________________________________
login_nodes               script                    /tmp/github.com/buildtesters/buildtest-cori/system/ping_nodes.yml
data_transfer_nodes       script                    /tmp/github.com/buildtesters/buildtest-cori/system/ping_nodes.yml
nerschost                 script                    /tmp/github.com/buildtesters/buildtest-cori/system/nerschost.yml
nameserver_ping           script                    /tmp/github.com/buildtesters/buildtest-cori/system/nameserver.yml
searchdomain_check        script                    /tmp/github.com/buildtesters/buildtest-cori/system/nameserver.yml
mount_check               script                    /tmp/github.com/buildtesters/buildtest-cori/system/mountpoint.yml
filesystem_access         script                    /tmp/github.com/buildtesters/buildtest-cori/system/filesystem.yml
cray_check                script                    /tmp/github.com/buildtesters/buildtest-cori/system/cray_env.yml
help                      script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/sqs.yml
summary                   script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/sqs.yml
jobs_by_partitions        script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/sqs.yml
show_running_jobs         script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/sqs.yml
show_nonrunning_jobs      script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/sqs.yml
show_jobs_qos             script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/sqs.yml
display_partition_info    script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/sqs.yml
jobstats                  script                    /tmp/github.com/buildtesters/buildtest-cori/slurmutils/jobstats.yml
current_user_queue        script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/squeue.yml
pending_jobs_shared_partition script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/squeue.yml
running_jobs_user_camelo  script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/squeue.yml
view_jobs_account_nstaff  script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/squeue.yml
sort_job_priority_shared_partition script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/squeue.yml
sinfo_version             script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/slurm_check.yml
nodes_state_down          script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
nodes_state_reboot        script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
nodes_state_allocated     script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
nodes_state_completing    script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
nodes_state_idle          script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
node_down_fail_list_reason script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
dead_nodes                script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
get_partitions            script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
drain_nodes_interactive_partition script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sinfo.yml
slurm_config              script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/scontrol.yml
show_node_config          script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/scontrol.yml
show_partition            script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/scontrol.yml
show_accounts             script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sacctmgr.yml
show_users                script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sacctmgr.yml
show_qos                  script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sacctmgr.yml
show_associations         script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sacctmgr.yml
show_tres                 script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/sacctmgr.yml
partition_check           script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/partition.yml
slurm_metadata            script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/valid_jobs/metadata.yml
knl_hostname              script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/valid_jobs/hostname.yml
haswell_hostname          script                    /tmp/github.com/buildtesters/buildtest-cori/slurm/valid_jobs/hostname.yml
iris_user_query           script                    /tmp/github.com/buildtesters/buildtest-cori/nersctools/iris.yml
iris_qos_gpu              script                    /tmp/github.com/buildtesters/buildtest-cori/nersctools/iris.yml
iris_project              script                    /tmp/github.com/buildtesters/buildtest-cori/nersctools/iris.yml
```

# Edit and View (Buildspecs)

You can use your editor (vi, vim, nano, emacs) to edit any file but if you want buildtest to validate your file interactively you
may find that ``buildtest buildspec edit`` to be useful.

Specify the name of buildspec you want based on output of ``buildtest buildspec find``. For example we can edit ``login_nodes`` which 
will open buildspec /tmp/github.com/buildtesters/buildtest-cori/system/ping_nodes.yml. This can be achieved by running:

```
buildtest buildspec edit login_nodes
```

When you invoke buildtest buildspec edit it will keep file open in a while loop until file
is valid for example we make some changes and save file and see the following warning. The user
can press a key and it will open file for editing until file is valid. You can bypass this by sending 
a control signal if you don't want to be stuck in endless loop.

```
$ buildtest buildspec edit login_nodes
version 1.1 is not known for type {'1.0': 'script-v1.0.schema.json', 'latest': 'script-v1.0.schema.json'}. Try using latest.
Press any key to continue
```

Since multiple tests can be defined in one YAML file you can pick any test name that belongs to the buildspec file path.

To view a buildspec use the ``buildtest buildspec view <name>`` to show content of the file. 

**Note: we show full content of buildspec and not each test so if multiple tests are found it will show all tests**

# Test Report 

You can run ``buildtest report`` to get report of all tests. Here is a preview output

```
$ buildtest report
name                 state                returncode           starttime            endtime              runtime              buildid              buildspec           
jobstats             PASS                 0                    2020/06/25 05:46:31  2020/06/25 05:46:43  011.25 jobstats_2020-06-25-05-46 /tmp/github.com/buildtesters/buildtest-cori/slurmutils/jobstats.yml
```

# Contributing Guide

To contribute back you will want to make sure your buildspec is validated before you contribute back, this could be 
done by running test manually ``buildtest build`` or see if buildspec is valid via ``buildtest buildspec find``.

buildtest relies on json schema to validate buildspecs and you will need to understand the json schema to understand how
to write valid tests. To get all schemas run the following
```
$ buildtest schema
script-v1.0.schema.json
compiler-v1.0.schema.json
global.schema.json
settings.schema.json
```

The schemas ``script``, ``compiler``, ``global`` are of interest when writing buildspec to view the json content for script 
schema you can run ``buildtest schema -n script-v1.0.schema.json --json``.

To view all examples for script schema you can run ``buildtest schema -n script-v1.0.schema.json --examples`` which will show
all invalid/valid tests. buildtest will validate each example before it shows content which will be help understand how to write
tests and see all the edge cases.

Check out the [contributing guide](https://buildtest.readthedocs.io/en/devel/contributing.html) for buildtest if you want to contribute back
to framework

# References

- buildtest documentation: https://buildtest.readthedocs.io/en/devel/
- buildtest schema docs: https://buildtesters.github.io/schemas/



