# Tutorial

To get started please login to Cori and [setup](README.md#setup) buildtest in your python environment. 

Once you are set, you can see preview of all tests in cache by run the folllowing command

```
$ buildtest buildspec find
```

You can see content of any test using `buildtest buildspec show`, shown below is output for test **shifter_job**.

```
(buildtest) siddiq90@cori03> buildtest buildspec show shifter_job
────────────────────── /global/u1/s/siddiq90/github/buildtest-cori/buildspecs/tools/shifter.yml ──────────────────────
╭───────────────────────────────────────────────────────────────────────────────────────────────────╮
│ version: "1.0"                                                                                    │
│ buildspecs:                                                                                       │
│   shifter_pull:                                                                                   │
│     type: script                                                                                  │
│     executor: cori.local.bash                                                                     │
│     description: Check if shifterimg pull can download a container                                │
│     tags: ["system", "containers"]                                                                │
│     run: shifterimg pull ubuntu:latest                                                            │
│                                                                                                   │
│   shifter_job:                                                                                    │
│     type: script                                                                                  │
│     executor: cori.slurm.haswell_debug                                                            │
│     description: Run shifter in a slurm job and run 32 instance of shifter image to echo hostname │
│     tags: ["containers", "jobs"]                                                                  │
│     sbatch: [-N 1, --image=docker:ubuntu:latest]                                                  │
│     run: srun -n 32 shifter hostname                                                              │
│                                                                                                   │
│ maintainers:                                                                                      │
│   - "shahzebsiddiqui"                                                                            │
│                                                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────╯
```

You can run this test as follows, this test will perform two tests, one will pull an image via shifter and run a shifter job in slurm job.

```
(buildtest) siddiq90@cori03> buildtest build -b buildspecs/tools/shifter.yml 
╭───────────────────────────────────────────────── buildtest summary ─────────────────────────────────────────────────╮
│                                                                                                                     │
│ User:               siddiq90                                                                                        │
│ Hostname:           cori10                                                                                          │
│ Platform:           Linux                                                                                           │
│ Current Time:       2022/07/01 06:26:39                                                                             │
│ buildtest path:     /global/homes/s/siddiq90/gitrepos/buildtest/bin/buildtest                                       │
│ buildtest version:  0.15.0                                                                                          │
│ python path:        /global/common/software/nersc/cori-2022q1/sw/python/3.9-anaconda-2021.11/bin/python3            │
│ python version:     3.9.7                                                                                           │
│ Configuration File: /global/u1/s/siddiq90/gitrepos/buildtest-nersc/config.yml                                       │
│ Test Directory:     /global/u1/s/siddiq90/gitrepos/buildtest/var/tests                                              │
│ Report File:        /global/u1/s/siddiq90/gitrepos/buildtest/var/report.json                                        │
│ Command:            /global/homes/s/siddiq90/gitrepos/buildtest/bin/buildtest build -b buildspecs/tools/shifter.yml │
│                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
────────────────────────────────────────────────────────────  Discovering Buildspecs ────────────────────────────────────────────────────────────
                             Discovered buildspecs
╔═════════════════════════════════════════════════════════════════════════════╗
║ buildspec                                                                   ║
╟─────────────────────────────────────────────────────────────────────────────╢
║ /global/u1/s/siddiq90/gitrepos/buildtest-nersc/buildspecs/tools/shifter.yml ║
╚═════════════════════════════════════════════════════════════════════════════╝


Total Discovered Buildspecs:  1
Total Excluded Buildspecs:  0
Detected Buildspecs after exclusion:  1
────────────────────────────────────────────────────────────── Parsing Buildspecs ───────────────────────────────────────────────────────────────
Buildtest will parse 1 buildspecs
Valid Buildspecs: 1
Invalid Buildspecs: 0
/global/u1/s/siddiq90/gitrepos/buildtest-nersc/buildspecs/tools/shifter.yml: VALID
Total builder objects created: 2
Total compiler builder: 0
Total script builder: 2
Total spack builder: 0
                                                             Script Builder Details
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ builder               ┃ executor                 ┃ compiler ┃ nodes ┃ procs ┃ description                    ┃ buildspecs                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ shifter_pull/97a1e57c │ cori.local.bash          │ None     │ None  │ None  │ Check if shifterimg pull can   │ /global/u1/s/siddiq90/gitrepo… │
│                       │                          │          │       │       │ download a container           │                                │
├───────────────────────┼──────────────────────────┼──────────┼───────┼───────┼────────────────────────────────┼────────────────────────────────┤
│ shifter_job/7c779344  │ cori.slurm.haswell_debug │ None     │ None  │ None  │ Run shifter in a slurm job and │ /global/u1/s/siddiq90/gitrepo… │
│                       │                          │          │       │       │ run 32 instance of shifter     │                                │
│                       │                          │          │       │       │ image to echo hostname         │                                │
└───────────────────────┴──────────────────────────┴──────────┴───────┴───────┴────────────────────────────────┴────────────────────────────────┘
                                                       Batch Job Builders
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ builder              ┃ executor                 ┃ buildspecs                                                                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ shifter_job/7c779344 │ cori.slurm.haswell_debug │ /global/u1/s/siddiq90/gitrepos/buildtest-nersc/buildspecs/tools/shifter.yml │
└──────────────────────┴──────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
───────────────────────────────────────────────────────────────── Building Test ─────────────────────────────────────────────────────────────────
shifter_pull/97a1e57c: Creating test directory: /global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/97a1e57c
shifter_pull/97a1e57c: Creating the stage directory:
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/97a1e57c/stage
shifter_pull/97a1e57c: Writing build script:
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/97a1e57c/shifter_pull_build.sh
shifter_job/7c779344: Creating test directory:
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.slurm.haswell_debug/shifter/shifter_job/7c779344
shifter_job/7c779344: Creating the stage directory:
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.slurm.haswell_debug/shifter/shifter_job/7c779344/stage
shifter_job/7c779344: Writing build script:
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.slurm.haswell_debug/shifter/shifter_job/7c779344/shifter_job_build.sh
───────────────────────────────────────────────────────────────── Running Tests ─────────────────────────────────────────────────────────────────
Spawning 64 processes for processing builders
────────────────────────────────────────────────────────────────── Iteration 1 ──────────────────────────────────────────────────────────────────
shifter_job/7c779344 does not have any dependencies adding test to queue
shifter_pull/97a1e57c does not have any dependencies adding test to queue
In this iteration we are going to run the following tests: [shifter_job/7c779344, shifter_pull/97a1e57c]
shifter_pull/97a1e57c: Running Test via command: bash --norc --noprofile -eo pipefail shifter_pull_build.sh
shifter_job/7c779344: Running Test via command: bash --norc --noprofile -eo pipefail shifter_job_build.sh
shifter_job/7c779344: JobID 60694501 dispatched to scheduler
shifter_pull/97a1e57c: Test completed in 5.154386 seconds
shifter_pull/97a1e57c: Test completed with returncode: 0
shifter_pull/97a1e57c: Writing output file -
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/97a1e57c/shifter_pull.out
shifter_pull/97a1e57c: Writing error file -
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/97a1e57c/shifter_pull.err
Polling Jobs in 30 seconds
shifter_job/7c779344: Job 60694501 is complete!
shifter_job/7c779344: Test completed in 35.205025 seconds
shifter_job/7c779344: Test completed with returncode: 0
shifter_job/7c779344: Writing output file -
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.slurm.haswell_debug/shifter/shifter_job/7c779344/shifter_job.out
shifter_job/7c779344: Writing error file -
/global/u1/s/siddiq90/gitrepos/buildtest/var/tests/cori.slurm.haswell_debug/shifter/shifter_job/7c779344/shifter_job.err
                                    Completed Jobs
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ builder              ┃ executor                 ┃ jobid    ┃ jobstate  ┃ runtime   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ shifter_job/7c779344 │ cori.slurm.haswell_debug │ 60694501 │ COMPLETED │ 35.205025 │
└──────────────────────┴──────────────────────────┴──────────┴───────────┴───────────┘
                                                        Test Summary
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ builder               ┃ executor                 ┃ status ┃ checks (ReturnCode, Regex, Runtime) ┃ returnCode ┃ runtime   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ shifter_job/7c779344  │ cori.slurm.haswell_debug │ PASS   │ N/A N/A N/A                         │ 0          │ 35.205025 │
├───────────────────────┼──────────────────────────┼────────┼─────────────────────────────────────┼────────────┼───────────┤
│ shifter_pull/97a1e57c │ cori.local.bash          │ PASS   │ N/A N/A N/A                         │ 0          │ 5.154386  │
└───────────────────────┴──────────────────────────┴────────┴─────────────────────────────────────┴────────────┴───────────┘



Passed Tests: 2/2 Percentage: 100.000%
Failed Tests: 0/2 Percentage: 0.000%


Adding 2 test results to /global/u1/s/siddiq90/gitrepos/buildtest/var/report.json
Writing Logfile to: /global/u1/s/siddiq90/gitrepos/buildtest/var/logs/buildtest_w3eiqmti.log
```

We can inspect the output of the test via `buildtest inspect query` with `-o` we fetch the output file. 

```
(buildtest) siddiq90@cori03> buildtest inspect query -o shifter_pull
───────────────────────────────── shifter_pull/93905ceb-3547-4f27-aa94-9f02b45bb4a9 ──────────────────────────────────
executor:  cori.local.bash
description:  Check if shifterimg pull can download a container
state:  PASS
returncode:  0
runtime:  4.417946
starttime:  2021/11/12 13:39:51
endtime:  2021/11/12 13:39:55
─ Output File: /global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/93905ceb/shift… ─
2021-11-12T13:39:51 Pulling Image: docker:ubuntu:latest, status: PULLING                                           
2021-11-12T13:39:52 Pulling Image: docker:ubuntu:latest, status: PULLING                                           
2021-11-12T13:39:52 Pulling Image: docker:ubuntu:latest, status: PULLING                                           
2021-11-12T13:39:53 Pulling Image: docker:ubuntu:latest, status: PULLING                                           
2021-11-12T13:39:53 Pulling Image: docker:ubuntu:latest, status: PULLING                                           
2021-11-12T13:39:54 Pulling Image: docker:ubuntu:latest, status: PULLING                                           
2021-11-12T13:39:55 Pulling Image: docker:ubuntu:latest, status: PULLING                                           
2021-11-12T13:39:55 Pulling Image: docker:ubuntu:latest, status: READY                                                                                                                       
```

The `buildtest path` command can be used to retrieve path to test, output, error file, etc... For this test we can fetch output file via

```
(buildtest) siddiq90@cori03> buildtest path -o shifter_pull
/global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/93905ceb/shifter_pull.out
```

You can use this to do more elaborate command such as 

```
cat $(buildtest path -o shifter_pull) | grep [PATTERN]
```

Let's try querying the buildspec cache and see all tests corresponding to tagname `e4s`, this can be done using the `--filter` option which filters buildspec cache. Try running the following command and see available tests:

```
buildtest buildspec find --filter tags=e4s
```

Let's take a look at test `e4s_22.02_dev_workflow` try running `buildtest buildspec show e4s_22.02_dev_workflow` to see content of test. This test will validate the spack developer workflow which we have outlined in the user documentation. Let's run this test by 
running the following command:

```
buildtest build -b buildspecs/apps/e4s/22.02/developer_workflow.yml
```

Upon completion take a look at output you can fetch output file using `buildtest path -o` which will return the filepath. Take note of the output, this test will install `papi` via spack in a spack environment defined by **spack.yaml**.


```
(buildtest) siddiq90@cori03> cat $(buildtest path -o e4s_22.02_dev_workflow)          
Creating python virtual environment in /global/homes/s/siddiq90/.spack-pyenv
Package                     Version
--------------------------- -------
cffi                        1.15.0
clingo                      5.5.2
pip                         21.2.3
pycparser                   2.21
python3-nersc-customs       0.4.0
python3-nersc-modster       0.4.0
python3-nersc-sitecustomize 0.4.0
python3-nerscjson           0.9.0
setuptools                  57.4.0
Your python interpreter used by spack is /global/homes/s/siddiq90/.spack-pyenv/bin/python
==> Updating view at /global/homes/s/siddiq90/spack-demo/.spack-env/view
==> Created environment in /global/homes/s/siddiq90/spack-demo
==> You can activate this environment with:
==>   spack env activate /global/homes/s/siddiq90/spack-demo
==> In environment /global/homes/s/siddiq90/spack-demo
==> Starting concretization pool with 2 processes
==> Environment concretized in 10.48 seconds.
==> Concretized papi%gcc
[+]  wph5r5w  papi@6.0.0.1%gcc@11.2.0~cuda+example~infiniband~lmsensors~nvml~powercap~rapl~rocm~rocm_smi~sde+shared~static_tools arch=cray-cnl7-haswell
==> Concretized papi%intel
[+]  5mdo5ba  papi@6.0.0.1%intel@19.1.2.254~cuda+example~infiniband~lmsensors~nvml~powercap~rapl~rocm~rocm_smi~sde+shared~static_tools arch=cray-cnl7-haswell
==> Installing environment /global/homes/s/siddiq90/spack-demo
==> All of the packages are already installed
==> Updating view at /global/homes/s/siddiq90/spack-demo/.spack-env/view
==> In environment /global/homes/s/siddiq90/spack-demo
==> Root specs
-- no arch / gcc ------------------------------------------------
-------------------------------- papi%gcc
-- no arch / intel ----------------------------------------------
-------------------------------- papi%intel
-- cray-cnl7-haswell / gcc@11.2.0 -------------------------------
wph5r5ws5zttcphffedwiywbh6tq4nkw papi@6.0.0.1~cuda+example~infiniband~lmsensors~nvml~powercap~rapl~rocm~rocm_smi~sde+shared~static_tools  /global/homes/s/siddiq90/spack-workspace/cori/software/cray-cnl7-haswell/gcc-11.2.0/papi-6.0.0.1-wph5r5ws5zttcphffedwiywbh6tq4nkw
-- cray-cnl7-haswell / intel@19.1.2.254 -------------------------
5mdo5bakcsj55qc24o22g3mpkjwvh4bk papi@6.0.0.1~cuda+example~infiniband~lmsensors~nvml~powercap~rapl~rocm~rocm_smi~sde+shared~static_tools  /global/homes/s/siddiq90/spack-workspace/cori/software/cray-cnl7-haswell/intel-19.1.2.254/papi-6.0.0.1-5mdo5bakcsj55qc24o22g3mpkjwvh4bk
==> Regenerating tcl module files
papi-6.0.0.1-gcc-11.2.0-wph5r5w
papi-6.0.0.1-intel-19.1.2.254-5mdo5ba
```


Let's push our test results to cdash via `buildtest cdash upload` and specify an arbitrary build name of your choice. For now we will pick the buildname `nersc-demo`. CDASH will group tests by build name. buildtest
will take all test results from the report file and upload them to CDASH and provide you a link to view results.

```
(buildtest) siddiq90@cori03> buildtest cdash upload nersc-demo
Reading configuration file:  /global/u1/s/siddiq90/github/buildtest-cori/config.yml
Reading report file:  /global/u1/s/siddiq90/github/buildtest/var/report.json
build name:  nersc-demo
site:  cori
stamp:  20211021-1340-Experimental
MD5SUM: c10346f095ee074c718e08eda06abfe1
PUT STATUS: 200
You can view the results at: https://my.cdash.org/viewTest.php?buildid=2090994
```

The site configuration includes the cdash setting which is used by buildtest when uploading the tests via `buildtest cdash upload`. The CDASH project is located at https://my.cdash.org/index.php?project=buildtest-cori. The `site: cori` is used to group test by different HPC systems (`cori`, `perlmutter`). 

```
    cdash:
      url: https://my.cdash.org
      project: buildtest-nersc
      site: cori
```

## Writing a buildspec

In this example, we will try writing a buildspec that will check the system git version and apply regular expression as criteria for success.

We can check that our system git version on Cori by running the following command

```
(buildtest) siddiq90@cori03> /usr/bin/git --version  
git version 2.26.2
```

Now let's create a buildspec which will run this same test. To do this please do the following

1. Create a file named `system_git.yml`
2. Name of test is `git_version` 
3. Use `type: script` to validate buildspec with script schema.
4. Test should run with executor `cori.local.bash`

If you did this correctly you should see the following

```
(buildtest) siddiq90@cori03> cat system_git.yml
version: "1.0"
buildspecs:
  git_version:
    type: script
    executor: cori.local.bash
    run: /usr/bin/git --version

```

Now let's run this test and inspect the output to see it looks correct.

```
buildtest build -b system_git.yml
buildtest inspect query -o git_version
```

Next we will apply a regular expression via `regex` property to ensure this test will pass based on version. First we will try to make regular expression that will make this test PASS and then tweak the expression
to see if it fails. This will ensure our test will change state incase `/usr/bin/git` gets upgraded to newer version. 

This can be done as follows. The regular expression is applied to `stdout` stream.

```yaml
version: "1.0"
buildspecs:
  git_version:
    type: script
    executor: cori.local.bash
    run: /usr/bin/git --version
    status:
      regex:
        stream: stdout
        exp: 'git version (2.26.2)'
```

Please rerun the same test and see if test `PASS` then update the expression to different value and rebuild the test and see if test FAILs.

```
exp: 'git version (2.26.1)'
```


