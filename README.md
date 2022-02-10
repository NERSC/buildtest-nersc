# buildtest-nersc

This repo is testsuite for Cori and Perlmutter system at NERSC using buildtest. A mirror of this repository is located at https://github.com/buildtesters/buildtest-nersc that is public facing. 


## Setup

To get started clone this repo and buildtest in your filesystem:

```
git clone https://github.com/buildtesters/buildtest.git
git clone https://software.nersc.gov/NERSC/buildtest-nersc.git
```

You will need python 3.7 or higher to [install buildtest](https://buildtest.readthedocs.io/en/devel/installing_buildtest.html), on Cori this can be done by loading python module
and create a conda environment as shown below. 

```
module load python/3.8-anaconda-2020.11
conda create -n buildtest
conda activate buildtest
source /path/to/buildtest/setup.sh
```

The perlmutter setup would be very similar just perform the same step with the default python module that comes with anaconda - `python/3.9-anaconda-2021.11`

Next, navigate to `buildtest-nersc` directory and set environment `BUILDTEST_CONFIGFILE` to point to **config.yml** which is the configuration file for NERSC system. 

```
cd buildtest-nersc
export BUILDTEST_CONFIGFILE=$(pwd)/config.yml
```

You can view and validate your configuration and see if configuration makes sense:

```
buildtest config validate
buildtest config view
```

Please make sure you are using tip of [devel](https://github.com/buildtesters/buildtest/tree/devel) branch of buildtest when writing tests. You should sync your local devel branch with upstream
fork, for more details see [contributing guide](https://buildtest.readthedocs.io/en/devel/contributing/code_contribution_guide.html).

First time around you should discover all buildspecs this can be done via ``buildtest buildspec find``.  The command below will find
and validate all buildspecs in the **buildtest-nersc** repo and load them in buildspec cache. Note that one needs to specify `--root` to specify location where
all buildspecs are located, we have not configured [buildspec_root](https://buildtest.readthedocs.io/en/devel/configuring_buildtest/overview.html#buildspec-roots) in the configuration file since we don't have a central location where this repo will reside.

```
buildtest buildspec find --root /path/to/buildtest-nersc/buildspecs --rebuild
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

You can see breakdown of tags and buildspec summary with the following commands

```
buildtest buildspec summary
buildtest buildspec find --group-by-tags
```

## Querying Tests

You can use `buildtest report` and `buildtest inspect` to query tests. The commands differ slightly and data is 
represented differently. The `buildtest report` command will show output in tabular form and only show some of the metadata,
if you want to access the entire test record use `buildtest inspect` command which displays the content in JSON format.
For more details on querying tests see https://buildtest.readthedocs.io/en/devel/gettingstarted/query_test_report.html


## CI Setup


Tests are run on schedule basis with one schedule corresponding to one gitlab job in [.gitlab-ci.yml](https://software.nersc.gov/NERSC/buildtest-nersc/-/blob/devel/.gitlab-ci.yml). The scheduled pipelines are configured in 
https://software.nersc.gov/NERSC/buildtest-nersc/-/pipeline_schedules. Each schedule has a variable ``TESTNAME`` defined to control which pipeline 
is run since we have multiple gitlab jobs. In the `.gitlab-ci.yml` we make use of conditional rules using [only](https://docs.gitlab.com/ee/ci/yaml/#onlyexcept-basic). 

The scheduled jobs are run at different intervals (1x/day, 1x/week, etc...) at different times of day to avoid overloading the system. The gitlab jobs
will run jobs based on tags, alternately some tests may be defined by running all tests in a directory (`buildtest build -b apps`). If you want to add a new
scheduled job, please define a [new schedule](https://software.nersc.gov/NERSC/buildtest-nersc/-/pipeline_schedules/new) with an appropriate time. The 
`target branch` should be `devel` and define a unique variable used to distinguish scheduled jobs. Next, create a job in `.gitlab-ci.yml` that references the scheduled job and define variable ``TESTNAME`` in the scheduled pipeline.


### Runner settings

This project is using a custom runner hosted via user [e4s](https://iris.nersc.gov/user/93315/info) in order for all jobs to be run by user *e4s*. The **e4s** user is a [collaboration account](https://docs.nersc.gov/accounts/collaboration_accounts/), if you don't have access to collaboration account, please contact the PI for project [m3503](https://iris.nersc.gov/project/67107/info) to see if you can be added to group.  If you are logged in to Cori/Perlmutter you can run the following command to switch to e4s user.

```
collabsu e4s
```

You will be prompted to type your NERSC password for your user account.

The `e4s` user has two shell runners configured with this project. Please note that you must be on the login node to the appropriate system to restart the runner

| System |  Runner Tag Name |
| ------- | ------------------- | 
| Cori | `tags: [cori-e4s]` |
| Perlmutter | `tags: [perlmutter-e4s]` | 

The runner can be started manually by running the following. 

```
# Cori
bash $HOME/cron/restart-cori.sh

# Perlmutter
bash $HOME/cron/restart-perlmutter.sh
```

The gitlab-runner is the ECP fork runner which can be found at `$HOME/nersc-ecp-staff-runner/`, and `gitlab-runner` should be in `$PATH` when you login via *e4s* which is set in `~/.bashrc` file.

The runner is tied to a particular hostname and is not run as a service, it is advisable to check/be notified when the runner goes down. There is a cronjob setup on `cori01` to automatically restart the gitlab runner if it goes down. Shown below is the crontab

```
# Crontab on Cori - cori01
e4s@cori01:~/cron> crontab -l
0 * * * * $HOME/cron/restart-cori.sh >/dev/null  2>&1 
```


All scheduled pipelines are run via `e4s` user your gitlab job must specify the appropriate tag name. You can check registered runner at https://software.nersc.gov/NERSC/buildtest-nersc/-/settings/ci_cd under `Runners` section. Please make sure `e4s` user has access to all the queues required to run tests, this can be configured in [iris](https://iris.nersc.gov/).

## Integrations

This project has integration with Slack to notify CI builds to [buildtest Slack](https://hpcbuildtest.slack.com) at **#cori-testsuite** workspace. The integrations can be 
found at https://software.nersc.gov/NERSC/buildtest-nersc/-/settings/integrations.

This project has setup a push mirror to https://github.com/buildtesters/buildtest-nersc which can be seen at https://software.nersc.gov/NERSC/buildtest-nersc/-/settings/repository 
under **Mirroring Repositories**. If the push mirror is not setup, please add the mirror. 

## CDASH

buildtest will push test results to [CDASH](https://www.kitware.com/cdash/project/about.html) server 
at https://my.cdash.org/index.php?project=buildtest-nersc using `buildtest cdash upload` command.

## Contributing Guide

To contribute back you will want to make sure your buildspec is validated before you contribute back, this could be 
done by running test manually `buildtest build` or see if buildspec is valid via `buildtest buildspec find`. It 
would be good to run your test and make sure it is working as expected, you can view test detail using `buildtest inspect name <testname>` or `buildtest inspect query <testname>`. For more 
details on querying test please see https://buildtest.readthedocs.io/en/devel/gettingstarted/query_test_report.html. 

If you want to contribute your tests, please see [CONTRIBUTING.md](https://software.nersc.gov/NERSC/buildtest-nersc/-/blob/devel/CONTRIBUTING.md)


## References

- buildtest documentation: https://buildtest.readthedocs.io/en/devel/
- buildtest schema docs: https://buildtesters.github.io/buildtest/
- Getting Started: https://buildtest.readthedocs.io/en/devel/getting_started.html
- Writing Buildspecs: https://buildtest.readthedocs.io/en/devel/buildspec_tutorial.html
- Contributing Guide: https://buildtest.readthedocs.io/en/devel/contributing.html


