# Tutorial

To get started please login to Cori and [setup](README.md#setup) buildtest in your python environment. 

Once you are set, you can see preview of all tests in cache by run the folllowing command

```
$ buildtest buildspec find
```

You can see content of any test using `buildtest buildspec show` and name of the test. 

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
╭──────────────────────────────────────────────── buildtest summary ────────────────────────────────────────────────╮ 
│                                                                                                                   │ 
│ User:               siddiq90                                                                                      │ 
│ Hostname:           cori03                                                                                        │ 
│ Platform:           Linux                                                                                         │ 
│ Current Time:       2021/11/12 13:39:51                                                                           │ 
│ buildtest path:     /global/homes/s/siddiq90/github/buildtest/bin/buildtest                                       │ 
│ buildtest version:  0.11.0                                                                                        │ 
│ python path:        /usr/common/software/python/3.8-anaconda-2020.11/bin/python                                   │ 
│ python version:     3.8.5                                                                                         │ 
│ Configuration File: /global/u1/s/siddiq90/github/buildtest-cori/config.yml                                        │ 
│ Test Directory:     /global/u1/s/siddiq90/github/buildtest/var/tests                                              │ 
│ Command:            /global/homes/s/siddiq90/github/buildtest/bin/buildtest build -b buildspecs/tools/shifter.yml │ 
│                                                                                                                   │ 
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯ 
──────────────────────────────────────────────  Discovering Buildspecs ───────────────────────────────────────────────
Discovered Buildspecs:  1
Excluded Buildspecs:  0
Detected Buildspecs after exclusion:  1
                           Discovered buildspecs                            
╔══════════════════════════════════════════════════════════════════════════╗
║ Buildspecs                                                               ║
╟──────────────────────────────────────────────────────────────────────────╢
║ /global/u1/s/siddiq90/github/buildtest-cori/buildspecs/tools/shifter.yml ║
╚══════════════════════════════════════════════════════════════════════════╝
───────────────────────────────────────────────── Parsing Buildspecs ─────────────────────────────────────────────────
Valid Buildspecs: 1
Invalid Buildspecs: 0
/global/u1/s/siddiq90/github/buildtest-cori/buildspecs/tools/shifter.yml: VALID


Total builder objects created: 2


                                                   Builder Details                                                    
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Builder               ┃ Executor                 ┃ description                    ┃ buildspecs                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ shifter_pull/93905ceb │ cori.local.bash          │ Check if shifterimg pull can   │ /global/u1/s/siddiq90/github/b │
│                       │                          │ download a container           │ uildtest-cori/buildspecs/tools │
│                       │                          │                                │ /shifter.yml                   │
├───────────────────────┼──────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ shifter_job/5fde86aa  │ cori.slurm.haswell_debug │ Run shifter in a slurm job and │ /global/u1/s/siddiq90/github/b │
│                       │                          │ run 32 instance of shifter     │ uildtest-cori/buildspecs/tools │
│                       │                          │ image to echo hostname         │ /shifter.yml                   │
└───────────────────────┴──────────────────────────┴────────────────────────────────┴────────────────────────────────┘
─────────────────────────────────────────────────── Building Test ────────────────────────────────────────────────────
[13:39:51] shifter_pull/93905ceb: Creating test directory:                                                 base.py:437
           /global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/93905ceb             
           shifter_pull/93905ceb: Creating the stage directory: /global/u1/s/siddiq90/github/buildtest/var base.py:448
           /tests/cori.local.bash/shifter/shifter_pull/93905ceb/stage                                                 
           shifter_pull/93905ceb: Writing build script: /global/u1/s/siddiq90/github/buildtest/var/tests/c base.py:565
           ori.local.bash/shifter/shifter_pull/93905ceb/shifter_pull_build.sh                                         
           shifter_job/5fde86aa: Creating test directory: /global/u1/s/siddiq90/github/buildtest/var/tests base.py:437
           /cori.slurm.haswell_debug/shifter/shifter_job/5fde86aa                                                     
           shifter_job/5fde86aa: Creating the stage directory: /global/u1/s/siddiq90/github/buildtest/var/ base.py:448
           tests/cori.slurm.haswell_debug/shifter/shifter_job/5fde86aa/stage                                          
           shifter_job/5fde86aa: Writing build script: /global/u1/s/siddiq90/github/buildtest/var/tests/co base.py:565
           ri.slurm.haswell_debug/shifter/shifter_job/5fde86aa/shifter_job_build.sh                                   
─────────────────────────────────────────────────── Running Tests ────────────────────────────────────────────────────
______________________________
Launching test: shifter_pull/93905ceb
shifter_pull/93905ceb: Running Test script 
/global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/93905ceb/shifter_pull_build.sh
______________________________
Launching test: shifter_job/5fde86aa
shifter_job/5fde86aa: Running Test script /global/u1/s/siddiq90/github/buildtest/var/tests/cori.slurm.haswell_debug/sh
ifter/shifter_job/5fde86aa/shifter_job_build.sh
shifter_pull/93905ceb: completed with returncode: 0
shifter_pull/93905ceb: Writing output file -  
/global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/93905ceb/shifter_pull.out
shifter_pull/93905ceb: Writing error file - 
/global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/93905ceb/shifter_pull.err
shifter_job/5fde86aa: JobID 50197716 dispatched to scheduler
Polling Jobs in 30 seconds
shifter_job/5fde86aa: Job 50197716 is complete! 
shifter_job/5fde86aa: Writing output file -  
/global/u1/s/siddiq90/github/buildtest/var/tests/cori.slurm.haswell_debug/shifter/shifter_job/5fde86aa/shifter_job.out
shifter_job/5fde86aa: Writing error file - 
/global/u1/s/siddiq90/github/buildtest/var/tests/cori.slurm.haswell_debug/shifter/shifter_job/5fde86aa/shifter_job.err
                   Pending Jobs                    
┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Builder ┃ executor ┃ JobID ┃ JobState ┃ runtime ┃
┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
└─────────┴──────────┴───────┴──────────┴─────────┘
                                    Completed Jobs                                    
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Builder              ┃ executor                 ┃ JobID    ┃ JobState  ┃ runtime   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ shifter_job/5fde86aa │ cori.slurm.haswell_debug │ 50197716 │ COMPLETED │ 35.184099 │
└──────────────────────┴──────────────────────────┴──────────┴───────────┴───────────┘
                                                     Test Summary                                                     
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Builder               ┃ executor                 ┃ status ┃ Checks (ReturnCode, Regex,    ┃ ReturnCode ┃ Runtime   ┃
┃                       ┃                          ┃        ┃ Runtime)                      ┃            ┃           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ shifter_pull/93905ceb │ cori.local.bash          │ PASS   │ N/A N/A N/A                   │ 0          │ 4.417946  │
├───────────────────────┼──────────────────────────┼────────┼───────────────────────────────┼────────────┼───────────┤
│ shifter_job/5fde86aa  │ cori.slurm.haswell_debug │ PASS   │ N/A N/A N/A                   │ 0          │ 35.184099 │
└───────────────────────┴──────────────────────────┴────────┴───────────────────────────────┴────────────┴───────────┘



Passed Tests: 2/2 Percentage: 100.000%
Failed Tests: 0/2 Percentage: 0.000%


Writing Logfile to: /tmp/buildtest_sr624mkc.log
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

The `buildtest path` command can be used to retrieve path to test, output, error file, etc... For this test we can fetch outputfile via

```
(buildtest) siddiq90@cori03> buildtest path -o shifter_pull
/global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.bash/shifter/shifter_pull/93905ceb/shifter_pull.out
```

You can use this to do more elaborate command such as 

```
cat $(buildtest path -o shifter_pull) | grep [PATTERN]
```

Let's run one of the E4S tests via `spack test` examples in buildtest. The e4s tests are tag with name `e4s` so you can filter output of `buildtest buildspec find` with tags as follows

```
buildtest buildspec find --filter tags=e4s
```

Let's take a look at the gasnet test we can see that this test will use [spack schema](https://buildtest.readthedocs.io/en/devel/buildspecs/spack.html) indicated by `type: spack` field to validate
the buildspec. This test will run against spec `gasnet@2021.3.0%intel` which is installed for `e4s/21.05` deployment.

```
(buildtest) siddiq90@cori03> buildtest buildspec show spack_test_gasnet_e4s_21.05
────────────────── /global/u1/s/siddiq90/github/buildtest-cori/buildspecs/e4s/spack_test/gasnet.yml ──────────────────
╭─────────────────────────────────────────────────────────────────────────────╮
│ version: "1.0"                                                              │
│ buildspecs:                                                                 │
│   spack_test_gasnet_e4s_21.05:                                              │
│     type: spack                                                             │
│     executor: cori.local.sh                                                 │
│     description: "Test gasnet@2021.3.0%intel with e4s/21.05 via spack test" │
│     tags: e4s                                                               │
│     pre_cmds: module load e4s/21.05                                         │
│     spack:                                                                  │
│       root: /global/common/software/spackecp/e4s-21.05/spack/               │
│       verify_spack: false                                                   │
│       test:                                                                 │
│         run:                                                                │
│           specs: ['gasnet@2021.3.0%intel']                                  │
│         results:                                                            │
│           option: '-l'                                                      │
│           specs: ['gasnet@2021.3.0%intel']                                  │
│ maintainers:                                                                │
│   - "shahzebsiddiqui"                                                       │
│   - "PHHargrove"                                                            │
│   - "bonachea"                                                              │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

Let's run this test

```
(buildtest) siddiq90@cori03> buildtest build -b buildspecs/e4s/spack_test/gasnet.yml
╭──────────────────────────────────────────────── buildtest summary ─────────────────────────────────────────────────╮
│                                                                                                                    │
│ User:               siddiq90                                                                                       │
│ Hostname:           cori03                                                                                         │
│ Platform:           Linux                                                                                          │
│ Current Time:       2021/11/12 13:47:57                                                                            │
│ buildtest path:     /global/homes/s/siddiq90/github/buildtest/bin/buildtest                                        │
│ buildtest version:  0.11.0                                                                                         │
│ python path:        /usr/common/software/python/3.8-anaconda-2020.11/bin/python                                    │
│ python version:     3.8.5                                                                                          │
│ Configuration File: /global/u1/s/siddiq90/github/buildtest-cori/config.yml                                         │
│ Test Directory:     /global/u1/s/siddiq90/github/buildtest/var/tests                                               │
│ Command:            /global/homes/s/siddiq90/github/buildtest/bin/buildtest build -b                               │
│ buildspecs/e4s/spack_test/gasnet.yml                                                                               │
│                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
──────────────────────────────────────────────  Discovering Buildspecs ───────────────────────────────────────────────
Discovered Buildspecs:  1
Excluded Buildspecs:  0
Detected Buildspecs after exclusion:  1
                               Discovered buildspecs                                
╔══════════════════════════════════════════════════════════════════════════════════╗
║ Buildspecs                                                                       ║
╟──────────────────────────────────────────────────────────────────────────────────╢
║ /global/u1/s/siddiq90/github/buildtest-cori/buildspecs/e4s/spack_test/gasnet.yml ║
╚══════════════════════════════════════════════════════════════════════════════════╝
───────────────────────────────────────────────── Parsing Buildspecs ─────────────────────────────────────────────────
Valid Buildspecs: 1
Invalid Buildspecs: 0
/global/u1/s/siddiq90/github/buildtest-cori/buildspecs/e4s/spack_test/gasnet.yml: VALID


Total builder objects created: 1


                                                   Builder Details                                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Builder                         ┃ Executor      ┃ description                    ┃ buildspecs                      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ spack_test_gasnet_e4s_21.05/904 │ cori.local.sh │ Test gasnet@2021.3.0%intel     │ /global/u1/s/siddiq90/github/bu │
│ 6f857                           │               │ with e4s/21.05 via spack test  │ ildtest-cori/buildspecs/e4s/spa │
│                                 │               │                                │ ck_test/gasnet.yml              │
└─────────────────────────────────┴───────────────┴────────────────────────────────┴─────────────────────────────────┘
─────────────────────────────────────────────────── Building Test ────────────────────────────────────────────────────
[13:47:57] spack_test_gasnet_e4s_21.05/9046f857: Creating test directory: /global/u1/s/siddiq90/github/bui base.py:437
           ldtest/var/tests/cori.local.sh/gasnet/spack_test_gasnet_e4s_21.05/9046f857                                 
           spack_test_gasnet_e4s_21.05/9046f857: Creating the stage directory: /global/u1/s/siddiq90/githu base.py:448
           b/buildtest/var/tests/cori.local.sh/gasnet/spack_test_gasnet_e4s_21.05/9046f857/stage                      
           spack_test_gasnet_e4s_21.05/9046f857: Writing build script: /global/u1/s/siddiq90/github/buildt base.py:565
           est/var/tests/cori.local.sh/gasnet/spack_test_gasnet_e4s_21.05/9046f857/spack_test_gasnet_e4s_2            
           1.05_build.sh                                                                                              
─────────────────────────────────────────────────── Running Tests ────────────────────────────────────────────────────
______________________________
Launching test: spack_test_gasnet_e4s_21.05/9046f857
spack_test_gasnet_e4s_21.05/9046f857: Running Test script /global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.
sh/gasnet/spack_test_gasnet_e4s_21.05/9046f857/spack_test_gasnet_e4s_21.05_build.sh
spack_test_gasnet_e4s_21.05/9046f857: completed with returncode: 0
spack_test_gasnet_e4s_21.05/9046f857: Writing output file -  /global/u1/s/siddiq90/github/buildtest/var/tests/cori.loc
al.sh/gasnet/spack_test_gasnet_e4s_21.05/9046f857/spack_test_gasnet_e4s_21.05.out
spack_test_gasnet_e4s_21.05/9046f857: Writing error file - /global/u1/s/siddiq90/github/buildtest/var/tests/cori.local
.sh/gasnet/spack_test_gasnet_e4s_21.05/9046f857/spack_test_gasnet_e4s_21.05.err
                                                     Test Summary                                                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Builder                         ┃ executor      ┃ status ┃ Checks (ReturnCode, Regex,     ┃ ReturnCode ┃ Runtime   ┃
┃                                 ┃               ┃        ┃ Runtime)                       ┃            ┃           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ spack_test_gasnet_e4s_21.05/904 │ cori.local.sh │ PASS   │ N/A N/A N/A                    │ 0          │ 38.853554 │
│ 6f857                           │               │        │                                │            │           │
└─────────────────────────────────┴───────────────┴────────┴────────────────────────────────┴────────────┴───────────┘



Passed Tests: 1/1 Percentage: 100.000%
Failed Tests: 0/1 Percentage: 0.000%


Writing Logfile to: /tmp/buildtest_yqs_ntfv.log
```

We can see the generated content of the test via `buildtest inspect query -t` followed by name of test. Its worth noting 
that `module load e4s/21.05` will drop us into a spack instance however the schema requires one to specify root of spack inorder
to `source` the setup script which is used for generating the line `source /global/common/software/spackecp/e4s-21.05/spack/share/spack/setup-env.sh` that 
initialize spack for e4s/21.05 stack.


```
(buildtest) siddiq90@cori03> buildtest inspect query -t spack_test_gasnet_e4s_21.05
────────────────────────── spack_test_gasnet_e4s_21.05/9046f857-d408-4503-a976-c72efb14ef3c ──────────────────────────
executor:  cori.local.sh
description:  Test gasnet@2021.3.0%intel with e4s/21.05 via spack test
state:  PASS
returncode:  0
runtime:  38.853554
starttime:  2021/11/12 13:47:57
endtime:  2021/11/12 13:48:36
─ Test File: /global/u1/s/siddiq90/github/buildtest/var/tests/cori.local.sh/gasnet/spack_test_gasnet_e4s_21.05/9046… ─
   1 #!/bin/bash                                                                                                      
   2                                                                                                                  
   3                                                                                                                  
   4 ######## START OF PRE COMMANDS ########                                                                          
   5 module load e4s/21.05                                                                                            
   6 ######## END OF PRE COMMANDS   ########                                                                          
   7                                                                                                                  
   8                                                                                                                  
   9 source /global/common/software/spackecp/e4s-21.05/spack/share/spack/setup-env.sh                                 
  10 spack test run --alias spack_test_gasnet_e4s_21.05 gasnet@2021.3.0%intel                                         
  11 spack test results -l -- gasnet@2021.3.0%intel                                                                   
```

We can confirm the output of this test by running the following command.

```
module load e4s/21.05
spack test results -l -- gasnet@2021.3.0%intel
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
      project: buildtest-cori
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


