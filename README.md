# buildtest-cori

This repository is [Cori](https://docs.nersc.gov/) Testsuite with buildtest. A mirror of this repository is located at https://gitlab.nersc.gov/nersc/consulting/buildtest-cori used for running gitlab CI specified in file [.gitlab-ci.yml](https://github.com/buildtesters/buildtest-cori/blob/devel/.gitlab-ci.yml).





## Setup

To get started clone this repo on Cori as follows:

```
git clone https://github.com/buildtesters/buildtest-cori
```

buildtest configuration file is read at `$HOME/.buildtest` if you don't have a directory please run

```
mkdir $HOME/.buildtest
```

Next, navigate to `buildtest-cori` directory and copy your site configuration to the following location

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

Please make sure you are using tip of `devel` with buildtest when writing tests. You should sync your local `devel` with upstream
fork, for more details see https://buildtest.readthedocs.io/en/devel/contributing/setup.html

First time around you should discover all buildspecs this can be done via ``buildtest buildspec find``.  The command below will find
and validate all buildspecs in the buildtest-cori repo and load them in buildspec cache.

```
$ buildtest buildspec find --root /path/to/buildtest-cori
```

The [**buildspecs_roots**](https://buildtest.readthedocs.io/en/devel/configuring_buildtest.html#buildspec-roots) property can be used to
define location where to search for buildspecs, this property is ommitted since you may clone this in arbitrary location. The ``--root`` option
is equivalent to specifying list of directories in `buildspecs_root` property defined in configuration file.

The buildspecs are loaded in buildspec cache file (JSON) that is used by `buildtest buildspec find` for querying cache. Subsequent runs will
read from cache.  For more details see [buildspec interface](https://buildtest.readthedocs.io/en/devel/getting_started.html#buildspecs-interface)


## Building Tests

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
can run the following:

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

## Test Report 

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

## CI Setup

Tests are run on schedule basis with one schedule corresponding to one gitlab job in `.gitlab-ci.yml`. The scheduled pipelines are configured in 
https://gitlab.nersc.gov/nersc/consulting/buildtest-cori/-/pipeline_schedules. Each schedule has a variable defined to control which pipeline 
is run. In the `.gitlab-ci.yml` we make use of conditional rules using [only](https://docs.gitlab.com/ee/ci/yaml/#onlyexcept-basic). For example the daily
system test has variable defined `DAILYCHECK` set to `True` and the gitlab job is defined as follows:


```
scheduled_system_check:
  stage: test
  only:
    refs:
      - schedules
    variables:
      - $DAILYCHECK == "True"
```

The scheduled jobs are run at different intervals (1x/day, 1x/week, etc...) at different times of day to avoid overloading the system. The gitlab jobs
will run jobs based on tags, alternately some tests may be defined by running all tests in a directory (`buildtest build -b apps`). If you want to add a new
scheduled job, please define a [New Schedule](https://gitlab.nersc.gov/nersc/consulting/buildtest-cori/-/pipeline_schedules/new) with an appropriate time. The `target branch` should be `devel` and define a unique variable used to distinguish scheduled jobs. Next, create a job in `.gitlab-ci.yml` that references the scheduled job based on the variable name.


The `validate_tests` gitlab job is responsible for validating buildspecs, please review this job when contributing tests. The buildspec must pass validation
in order for buildtest to build and run the test. 

### Register Runner

We have a custom runner to run pipelines via username `siddiq90` on Cori. This runner must be registered with gitlab in order for gitlab to submit jobs. The runner registration is done once, if runner is registered you can skip this section. 

In order to register runner, please switch to user `siddiq90` and navigate to the following directory `$HOME/.config/systemd/user`.

There will be a `gitlab-runner` program used for registering gitlab runners. To register a new runner interactively please run:

```
gitlab-runner register
```

This will ask you few prompt before runner is registered. For more details see [Registering Runners](https://docs.gitlab.com/runner/register/). 

In the event Cori is down due to system outage, the runner will need to be started again, this can be done by running `gitlab-runner run`.


## Integrations

buildtest-cori mirror repo has integration with Github and Slack. The integrations can be found at https://gitlab.nersc.gov/nersc/consulting/buildtest-cori/-/settings/integrations. The Github integration will push result back to upstream project: https://github.com/buildtesters/buildtest-cori. The CI checks are pushed to [NERSC Slack](https://nersc.slack.com) at **#buildtest** workspace. 


## Contributing Guide

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

If you want to contribute your tests, please send [CONTRIBUTING.md](https://github.com/buildtesters/buildtest-cori/blob/master/CONTRIBUTING.md)


## References

- buildtest documentation: https://buildtest.readthedocs.io/en/devel/
- buildtest schema docs: https://buildtesters.github.io/buildtest/
- Getting Started: https://buildtest.readthedocs.io/en/devel/getting_started.html
- Writing Buildspecs: https://buildtest.readthedocs.io/en/devel/writing_buildspecs.html
- Contributing Guide: https://buildtest.readthedocs.io/en/devel/contributing.html


