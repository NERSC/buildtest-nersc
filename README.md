# buildtest-cori

This repository is [Cori](https://docs.nersc.gov/) Testsuite with buildtest. A mirror of this repository is located at https://software.nersc.gov/siddiq90/buildtest-cori used for running gitlab CI specified in file [.gitlab-ci.yml](https://github.com/buildtesters/buildtest-cori/blob/devel/.gitlab-ci.yml).



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
cp config.yml $HOME/.buildtest/config.yml
```

Assuming you have [installed buildtest](https://buildtest.readthedocs.io/en/devel/installing_buildtest.html) you 
can view and validate your configuration via:

```
buildtest config validate
buildtest config view
```

Please make sure you are using tip of `devel` with buildtest when writing tests. You should sync your local `devel` with upstream
fork, for more details see https://buildtest.readthedocs.io/en/devel/contributing/code_contribution_guide.html

First time around you should discover all buildspecs this can be done via ``buildtest buildspec find``.  The command below will find
and validate all buildspecs in the buildtest-cori repo and load them in buildspec cache.

```
$ buildtest buildspec find --root /path/to/buildtest-cori/buildspecs --rebuild
```

The buildspecs are loaded in buildspec cache file (JSON) that is used by `buildtest buildspec find` for querying cache. Subsequent runs will
read from cache.  For more details see [buildspec interface](https://buildtest.readthedocs.io/en/devel/gettingstarted/buildspecs_interface.html).


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
can run: ``buildtest buildspec find --tags``


For instance if you want to run all ``lustre`` tests you can run the following:

```
buildtest build --tags lustre
```

For more details on buildtest test please see the [buildtest tutorial](https://buildtest.readthedocs.io/en/devel/getting_started.html)

## Tags Breakdown

When you write buildspecs, please make sure you attach one or more `tags` to the test that way your test will get picked up during one of the CI checks. Shown
below is a summary of tag description

- **daily** - this tag is used for running daily system checks using gitlab CI. Tests should run relatively quick
- **system** - this tag is used for classifying all system tests that may include: system configuration, servers, network, cray tests. This tag should be used 
- **slurm** - this tag is used for slurm test that includes slurm utility check, slurm controller, etc... This tag **shouldn't** be used for job submission that is managed by **jobs** tag. The `slurm` tag tests should be short running test that use a Local Executor.
- **jobs** - this tag is used for testing slurm policies by submitting jobs to scheduler. 
- **compile** - this tag is used for compilation of application (OpenMP, MPI, OpenACC, CUDA, upc, bupc, etc...)
- **e4s** - this tag is used for running tests for E4S stack via `spack test` or [E4S Testsuite](https://github.com/E4S-Project/testsuite).
- **module** - this tag is used for testing module system
- **benchmark** - this tag is used for benchmark tests. This can be application benchmarks, mini-benchmarks, kernels, etc... 

## Querying Tests

You can use ``buildtest report`` and ``buildtest inspect`` to query tests. The commands differ slightly and data is 
represented differently. The ``buildtest report`` command will show output in tabular form and only show some of the metadata,
if you want to access the entire test record use ``buildtest inspect`` command which displays the content in JSON format.
For more details on querying tests see https://buildtest.readthedocs.io/en/devel/gettingstarted/query_test_report.html


## CI Setup

There is a github workflow [.mirror_to_cori.yml](https://github.com/buildtesters/buildtest-cori/blob/devel/.github/workflows/mirror_to_cori.yml) 
responsible for mirroring upstream project to https://software.nersc.gov/siddiq90/buildtest-cori.  We have setup a gitlab 
[secret](https://github.com/buildtesters/buildtest-cori/settings/secrets/actions) that contains gitlab personal access token created from 
https://software.nersc.gov/-/profile/personal_access_tokens. The Personal access token must have `read_api`, `read_repository`, `write_repository` scope.  

Tests are run on schedule basis with one schedule corresponding to one gitlab job in [.gitlab-ci.yml](https://github.com/buildtesters/buildtest-cori/blob/devel/.gitlab-ci.yml). The scheduled pipelines are configured in 
https://software.nersc.gov/siddiq90/buildtest-cori/-/pipeline_schedules. Each schedule has a variable defined to control which pipeline 
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
scheduled job, please define a [new schedule](https://software.nersc.gov/siddiq90/buildtest-cori/-/pipeline_schedules/new) with an appropriate time. The 
`target branch` should be `devel` and define a unique variable used to distinguish scheduled jobs. Next, create a job in `.gitlab-ci.yml` that references 
the scheduled job based on the variable name.

### Runner settings

The runner on Cori is generally run on the hostname `cori01` by the user [e4s](https://iris.nersc.gov/user/93315/info). The runner can be started using the following command 
```
screen -dmS e4s-runner-buildtest $(which gitlab-runner) run
```

The command allows the process to detach from the terminal and remain active on logoff. The gitlab-runner in question is the ECP fork runner which can be found at `/global/homes/e/e4s/nersc-ecp-staff-runner/ecp-ci-0.6.1/gitlab-runner/out/binaries/gitlab-runner`. The config.toml is not specified in the command line because it is stored in the default config location `$HOME/.gitlab-runner/config.toml`.

The runner is tied to a particular hostname and is not run as a service, it is advisable to check/be notified when the runner goes down. It may be useful to run a cronjob to check the runner status.

```
kavaluav@cori01:~> crontab -l
#This cronjob checks whether the E4S runner is active
PATH=/bin:/usr/bin:/usr/local/bin
00 * * * * /bin/bash /global/cscratch1/sd/kavaluav/e4s_runner_check.sh > /dev/null 2>&1
```

The script referenced above has the below content.
```
kavaluav@cori01:~> cat /global/cscratch1/sd/kavaluav/e4s_runner_check.sh
#!/bin/bash

process=$(ps -u e4s | grep -i screen | wc -l) 
if [[ $process -lt 1 ]]
then
	mail -s "e4s runner is dead" akavalur@lbl.gov <<< "E4S runner on Cori is no longer visible"
fi

```

All scheduled pipelines are run via `e4s` user which is done by  specifying `tags: [c_e4s_cori01]` in gitlab job 
to run jobs with a single user account. Please make sure `e4s` user has access to all the queues required to run tests.
This can be configured in [iris](https://iris.nersc.gov/).

## Integrations

buildtest-cori mirror repo has integration with Github and Slack. The integrations can be found at 
https://software.nersc.gov/siddiq90/buildtest-cori/-/settings/integrations. The Github integration 
will push result back to upstream project: https://github.com/buildtesters/buildtest-cori. The CI
checks are pushed to [buildtest Slack](https://hpcbuildtest.slack.com) at **#cori-testsuite** workspace. 

## CDASH

buildtest will push test results to [CDASH](https://www.kitware.com/cdash/project/about.html) server 
at https://my.cdash.org/index.php?project=buildtest-cori using `buildtest cdash upload` command which 
uploads all tests found in report file found in **$HOME/.buildtest/report.json**. 

## Contributing Guide

To contribute back you will want to make sure your buildspec is validated before you contribute back, this could be 
done by running test manually ``buildtest build`` or see if buildspec is valid via ``buildtest buildspec find``. It 
would be good to run your test and make sure it is working as expected, you can view test detail using ``buildtest inspect id <ID>`` 
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

Alternately you can see all published schemas and examples on https://buildtest.readthedocs.io/en/devel/schema_examples.html

If you want to contribute your tests, please see [CONTRIBUTING.md](https://github.com/buildtesters/buildtest-cori/blob/devel/CONTRIBUTING.md)


## References

- buildtest documentation: https://buildtest.readthedocs.io/en/devel/
- buildtest schema docs: https://buildtesters.github.io/buildtest/
- Getting Started: https://buildtest.readthedocs.io/en/devel/getting_started.html
- Writing Buildspecs: https://buildtest.readthedocs.io/en/devel/buildspec_tutorial.html
- Contributing Guide: https://buildtest.readthedocs.io/en/devel/contributing.html


