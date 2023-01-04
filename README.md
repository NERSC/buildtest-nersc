# buildtest-nersc

This repository contains tests for Cori and Perlmutter using the [buildtest](https://buildtest.readthedocs.io/en/devel/) framework. 

## Useful Links

- CDASH: https://my.cdash.org/index.php?project=buildtest-nersc
- Upstream Repo: https://software.nersc.gov/NERSC/buildtest-nersc
- Github Mirror Repo: https://github.com/buildtesters/buildtest-nersc 

## Buildtest References

- Documentation: https://buildtest.readthedocs.io/en/devel/
- Schema Docs: https://buildtesters.github.io/buildtest/
- Slack Channel: https://hpcbuildtest.slack.com
- Getting Started: https://buildtest.readthedocs.io/en/devel/getting_started.html
- Writing Buildspecs: https://buildtest.readthedocs.io/en/devel/buildspec_tutorial.html
- Contributing Guide: https://buildtest.readthedocs.io/en/devel/contributing.html

## Setup

To get started, please [connect to NERSC system](https://docs.nersc.gov/connect/) and clone this repo and buildtest:

```
git clone https://github.com/buildtesters/buildtest
git clone https://software.nersc.gov/NERSC/buildtest-nersc
```

Note if you don't have access to Gitlab server you may clone the mirror on Github:

```
git clone https://github.com/buildtesters/buildtest-nersc
```

You will need python 3.7 or higher to [install buildtest](https://buildtest.readthedocs.io/en/devel/installing_buildtest.html), on Cori/Perlmutter this can be done by loading **python**
module and create a conda environment as shown below.

```
module load python
conda create -n buildtest
conda activate buildtest
```

Now let's install buildtest, assuming you have cloned buildtest in $HOME directory source the setup script. For csh users you need to source **setup.csh**

```
source ~/buildtest/setup.sh

# csh users
source ~/buildtest/setup.csh
```

Next, navigate to `buildtest-nersc` directory and set environment `BUILDTEST_CONFIGFILE` to point to [config.yml](https://software.nersc.gov/NERSC/buildtest-nersc/-/blob/devel/config.yml) which is the configuration file for NERSC system.

```
cd buildtest-nersc
export BUILDTEST_CONFIGFILE=$(pwd)/config.yml
```

Make sure the configuration is valid, this can be done by running the following. buildtest will validate the configuration file with the JSON schema :

```
buildtest config validate
```

Please make sure you are using tip of [devel](https://github.com/buildtesters/buildtest/tree/devel) branch of buildtest when writing tests. You should sync your local devel branch with upstream
fork, for more details see [contributing guide](https://buildtest.readthedocs.io/en/devel/contributing/code_contribution_guide.html).

First time around you should discover all buildspecs this can be done via ``buildtest buildspec find``.  The command below will find
and validate all buildspecs in the **buildtest-nersc** repo and load them in buildspec cache. Note that one needs to specify `--root` to specify location where
all buildspecs are located, we have not configured [buildspec_root](https://buildtest.readthedocs.io/en/devel/configuring_buildtest/overview.html#buildspec-roots) in the configuration file since we don't have a central location where this repo will reside.

```
cd buildtest-nersc
buildtest buildspec find --root buildspecs --rebuild -q
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

## Integrations

This project has integration with Slack to notify CI builds to [buildtest Slack](https://hpcbuildtest.slack.com) at **#buildtest-nersc** workspace. The integrations can be
found at https://software.nersc.gov/NERSC/buildtest-nersc/-/settings/integrations.

This project has setup a push mirror to https://github.com/buildtesters/buildtest-nersc which can be seen at https://software.nersc.gov/NERSC/buildtest-nersc/-/settings/repository
under **Mirroring Repositories**. If the push mirror is not setup, please add the mirror.

## CDASH

buildtest will push test results to [CDASH](https://www.cdash.org/) server
at https://my.cdash.org/index.php?project=buildtest-nersc using `buildtest cdash upload` command.

## Contributing Guide

To contribute back you will want to make sure your buildspec is validated before you contribute back, this could be
done by running test manually `buildtest build` or see if buildspec is valid via `buildtest buildspec find`. It
would be good to run your test and make sure it is working as expected, you can view test detail using `buildtest inspect name <testname>` or `buildtest inspect query <testname>`. For more
details on querying test please see https://buildtest.readthedocs.io/en/devel/gettingstarted/query_test_report.html.

If you want to contribute your tests, please see [CONTRIBUTING.md](https://software.nersc.gov/NERSC/buildtest-nersc/-/blob/devel/CONTRIBUTING.md)

## Submitting an Issue

Please submit all issues to https://github.com/buildtesters/buildtest-nersc/issues. When creating an issue, please see the [labels](https://github.com/buildtesters/buildtest-nersc/labels) 
and try to select one or more labels to categorize issue. Please use the following labels depending on the type of issue you are reporting

- [Bug](https://github.com/buildtesters/buildtest-nersc/labels/bug): When creating an issue related to a test bug
- [new-test](https://github.com/buildtesters/buildtest-nersc/labels/new-test): An issue for adding a new test
- [E4S-Testsuite](https://github.com/buildtesters/buildtest-nersc/labels/E4S-Testsuite): Issues related to [E4S testsuite project](https://github.com/E4S-Project/testsuite) 
- [spack](https://github.com/buildtesters/buildtest-nersc/labels/spack): Issues related to `spack test`
- [documentation](https://github.com/buildtesters/buildtest-nersc/labels/documentation): Issues with documentation such as README.md, CONTRIBUTING.md
- [gitlab-ci](https://github.com/buildtesters/buildtest-nersc/labels/gitlab-ci): Issues with Gitlab CI/CD

