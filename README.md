# buildtest-cori

This repository is [Cori](https://docs.nersc.gov/) Testsuite with buildtest. A mirror of this repository is located at https://gitlab.nersc.gov/nersc/consulting/buildtest-cori used for running CI checks. 



## Setup

To get started clone this repo.
```
git clone https://github.com/buildtesters/buildtest-cori
```

Next copy your site configuration to the following location

```
cd buildtest-cori
cp .buildtest/config.yml $HOME/.buildtest/config.yml
```

Assuming you have [installed buildtest](https://buildtest.readthedocs.io/en/devel/installing_buildtest.html) you 
can view and validate your configuration via:

```
buildtest config validate
buildtest config view
```

First time around you should discover all buildspecs this can be done via ``buildtest buildspec find``. Please consider checking 
[**buildspecs_roots**](https://buildtest.readthedocs.io/en/devel/configuring_buildtest.html#buildspec-roots) in your configuration
to root of buildtest-cori repo so you can discover Cori test. If you are able to get this far you should see a lot more tests.
The ``--rebuild`` will force rebuild your buildspec cache.

```

(buildtest) siddiq90@cori06:~/buildtest-cori/jobs> buildtest buildspec find --rebuild
Clearing cache file: /global/u1/s/siddiq90/buildtest/var/buildspec-cache.json
Found 123 buildspecs
Validated 15/123 buildspecs
Validated 20/123 buildspecs
Validated 25/123 buildspecs
Validated 30/123 buildspecs
Validated 35/123 buildspecs
Validated 40/123 buildspecs
Validated 45/123 buildspecs
Validated 50/123 buildspecs
Validated 55/123 buildspecs
Validated 60/123 buildspecs
Validated 65/123 buildspecs
Validated 70/123 buildspecs
Validated 75/123 buildspecs
Validated 80/123 buildspecs
Validated 85/123 buildspecs
Validated 90/123 buildspecs
Validated 95/123 buildspecs
Validated 100/123 buildspecs
Validated 105/123 buildspecs
Validated 110/123 buildspecs
Validated 115/123 buildspecs
Validated 120/123 buildspecs
Validated 123/123 buildspecs

Detected 4 invalid buildspecs 

Writing invalid buildspecs to file: /global/u1/s/siddiq90/buildtest/var/buildspec.error 



+---------------------------------+----------+---------------+------------------------------------------------------+----------------------------------------------------------------------------------+
| Name                            | Type     | Executor      | Tags                                                 | Description                                                                      |
+=================================+==========+===============+======================================================+==================================================================================+
| csh_shell                       | script   | local.csh     | ['tutorials']                                        | csh shell example                                                                |
+---------------------------------+----------+---------------+------------------------------------------------------+----------------------------------------------------------------------------------+
| python_hello                    | script   | local.bash    | python                                               | Hello World python                                                               |
+---------------------------------+----------+---------------+------------------------------------------------------+----------------------------------------------------------------------------------+
| selinux_disable                 | script   | local.bash    | ['tutorials']                                        | Check if SELinux is Disabled                                                     |
+---------------------------------+----------+---------------+------------------------------------------------------+----------------------------------------------------------------------------------+
| systemd_default_target          | script   | local.bash    | ['tutorials']                                        | check if default target is multi-user.target                                     |
+---------------------------------+----------+---------------+------------------------------------------------------+----------------------------------------------------------------------------------+
| circle_area                     | script   | local.python  | ['tutorials', 'python']                              | Calculate circle of area given a radius                                          |
+---------------------------------+----------+---------------+------------------------------------------------------+----------------------------------------------------------------------------------+
| hello_world                     | script   | local.bash    | tutorials                                            | hello world example                                                              |
+---------------------------------+----------+---------------+------------------------------------------------------+--------------------------------------------

...
```

# Building Tests

**Note: All tests are written in YAML using .yml extension**
   
To build tests use ``buildtest build`` command for example we build all tests in ``system`` directory as follows

```
buildtest build -b system/
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

buildtest can run tests via tags which can be useful when grouping tests, to see a list of available tags you 
can run the following

```
$ buildtest buildspec find --tags
+---------------+
| Tags          |
+===============+
| hourly        |
+---------------+
| pass          |
+---------------+
| slurm         |
+---------------+
| tool          |
+---------------+
| benchmark     |
+---------------+
| benchmarks    |
+---------------+
| containers    |
+---------------+
| python        |
+---------------+
| petsc         |
+---------------+
| cray          |
+---------------+
| network       |
+---------------+
| openmp        |
+---------------+
| reframe       |
+---------------+
| configuration |
+---------------+
| checkout      |
+---------------+
| gpu           |
+---------------+
| storage       |
+---------------+
| jobs          |
+---------------+
| gpfs          |
+---------------+
| spack         |
+---------------+
| lustre        |
+---------------+
| darshan       |
+---------------+
| lsf           |
+---------------+
| datawarp      |
+---------------+
| ping          |
+---------------+
| e4s           |
+---------------+
| tools         |
+---------------+
| system        |
+---------------+
| misc          |
+---------------+
| compile       |
+---------------+
| fail          |
+---------------+
| ssh           |
+---------------+
| queues        |
+---------------+
| esslurm       |
+---------------+
| mkl           |
+---------------+
| cvmfs         |
+---------------+
| filesystem    |
+---------------+
| modules       |
+---------------+
| mpi           |
+---------------+
| tutorials     |
+---------------+
| openacc       |
+---------------+
```

For instance if you want to run all ``lustre`` tests you can use ``buildtest build --tags`` option. 

```
   $ buildtest build --tags lustre

   +-------------------------------+
   | Stage: Discovering Buildspecs |
   +-------------------------------+ 


   Discovered Buildspecs:

   /global/u1/s/siddiq90/buildtest-cori/filesystem/lustre.yml

   +---------------------------+
   | Stage: Parsing Buildspecs |
   +---------------------------+ 

    schemafile              | validstate   | buildspec
   -------------------------+--------------+------------------------------------------------------------
    script-v1.0.schema.json | True         | /global/u1/s/siddiq90/buildtest-cori/filesystem/lustre.yml

   +----------------------+
   | Stage: Building Test |
   +----------------------+ 

    name                | id       | type   | executor   | tags                     | testpath
   ---------------------+----------+--------+------------+--------------------------+-----------------------------------------------------------------------------------------------------
    lustre_osts         | b487c36d | script | local.bash | ['filesystem', 'lustre'] | /global/u1/s/siddiq90/buildtest/var/tests/local.bash/lustre/lustre_osts/1/stage/generate.sh
    lustre_mdts         | 03259756 | script | local.bash | ['filesystem', 'lustre'] | /global/u1/s/siddiq90/buildtest/var/tests/local.bash/lustre/lustre_mdts/1/stage/generate.sh
    lustre_nstaff_quota | ad748ead | script | local.bash | ['filesystem', 'lustre'] | /global/u1/s/siddiq90/buildtest/var/tests/local.bash/lustre/lustre_nstaff_quota/1/stage/generate.sh

   +----------------------+
   | Stage: Running Test  |
   +----------------------+ 

    name                | id       | executor   | status   |   returncode | testpath
   ---------------------+----------+------------+----------+--------------+-----------------------------------------------------------------------------------------------------
    lustre_osts         | b487c36d | local.bash | PASS     |            0 | /global/u1/s/siddiq90/buildtest/var/tests/local.bash/lustre/lustre_osts/1/stage/generate.sh
    lustre_mdts         | 03259756 | local.bash | PASS     |            0 | /global/u1/s/siddiq90/buildtest/var/tests/local.bash/lustre/lustre_mdts/1/stage/generate.sh
    lustre_nstaff_quota | ad748ead | local.bash | PASS     |            0 | /global/u1/s/siddiq90/buildtest/var/tests/local.bash/lustre/lustre_nstaff_quota/1/stage/generate.sh

   +----------------------+
   | Stage: Test Summary  |
   +----------------------+ 

   Executed 3 tests
   Passed Tests: 3/3 Percentage: 100.000%
   Failed Tests: 0/3 Percentage: 0.000%


```
# Test Report 

You can run ``buildtest report`` to get report of all tests. Here is a preview output

```
$ buildtest report
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| name                            | id       | state   |   returncode | starttime           | endtime             |    runtime | tags                                 | buildspec                                                                    |
+=================================+==========+=========+==============+=====================+=====================+============+======================================+==============================================================================+
| xfer_qos_hostname               | a0b0f578 | PASS    |            0 | 2020-10-14T17:11:17 | 2020-10-14T17:11:18 |  0         | queues                               | /global/u1/s/siddiq90/buildtest-cori/queues/xfer.yml                         |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| xfer_qos_hostname               | d0043be3 | PASS    |            0 | 2020-10-16T14:21:50 | 2020-10-16T14:21:51 |  0         | queues                               | /global/u1/s/siddiq90/buildtest-cori/queues/xfer.yml                         |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| debug_qos_knl_hostname          | e8c32e83 | PASS    |            0 | 2020-10-14T17:11:17 | 2020-10-14T17:11:20 |  0         | queues reframe                       | /global/u1/s/siddiq90/buildtest-cori/queues/debug.yml                        |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| debug_qos_knl_hostname          | 4b84492d | PASS    |            0 | 2020-10-21T14:34:22 | 2020-10-21T14:34:29 |  0         | queues reframe                       | /global/u1/s/siddiq90/buildtest-cori/queues/debug.yml                        |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| debug_qos_haswell_hostname      | fe76218c | PASS    |            0 | 2020-10-14T17:11:17 | 2020-10-14T17:11:20 |  0         | queues reframe                       | /global/u1/s/siddiq90/buildtest-cori/queues/debug.yml                        |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| premium_qos_haswell_hostname    | 7a67ae00 | PASS    |            0 | 2020-10-14T17:11:23 | 2020-10-14T17:11:27 |  0         | queues reframe                       | /global/u1/s/siddiq90/buildtest-cori/queues/premium.yml                      |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| premium_qos_knl_hostname        | ef64cd6f | PASS    |            0 | 2020-10-14T17:11:23 | 2020-10-14T17:11:27 |  0         | queues reframe                       | /global/u1/s/siddiq90/buildtest-cori/queues/premium.yml                      |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
| bigmem_qos_hostname             | 03efbce4 | PASS    |            0 | 2020-10-14T17:11:20 | 2020-10-14T17:11:21 |  0         | queues reframe                       | /global/u1/s/siddiq90/buildtest-cori/queues/bigmem.yml                       |
+---------------------------------+----------+---------+--------------+---------------------+---------------------+------------+--------------------------------------+------------------------------------------------------------------------------+
```

# Contributing Guide

To contribute back you will want to make sure your buildspec is validated before you contribute back, this could be 
done by running test manually ``buildtest build`` or see if buildspec is valid via ``buildtest buildspec find``. It 
would be good to run your test and make sure it is working as expected, you can view test detail using ``buildtest inspect <ID>`` 
or see [Test Inspection](https://buildtest.readthedocs.io/en/devel/getting_started.html#test-inspection) section. 

buildtest relies on json schema to validate buildspecs and you will need to understand the json schema to understand how
to write valid tests. To get all schemas run the following:
```
$ buildtest schema
global.schema.json
definitions.schema.json
settings.schema.json
compiler-v1.0.schema.json
script-v1.0.schema.json
```

The schemas ``script``, ``compiler``, ``global`` are of interest when writing buildspec to view the json content for script 
schema you can run ``buildtest schema -n script-v1.0.schema.json --json``.

To view all examples for script schema you can run ``buildtest schema -n script-v1.0.schema.json --examples`` which will show
all invalid/valid tests. buildtest will validate each example before it shows content which will be help understand how to write
tests and see all the edge cases.

Alternately you can see all published schemas and examples on https://buildtest.readthedocs.io/en/devel/buildspecs/schema_examples.html

If you want to contribute your tests, please send a pull request to `devel` branch.


# References

- buildtest documentation: https://buildtest.readthedocs.io/en/devel/
- buildtest schema docs: https://buildtesters.github.io/buildtest/
- Getting Started: https://buildtest.readthedocs.io/en/devel/getting_started.html
- Writing Buildspecs: https://buildtest.readthedocs.io/en/devel/writing_buildspecs.html
- Contributing Guide: https://buildtest.readthedocs.io/en/devel/contributing.html


