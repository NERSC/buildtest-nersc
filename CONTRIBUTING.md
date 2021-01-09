Getting Started
================

Contribution is not easy, so we created this document to describe how to get you setup
so you can contribute back and make everyone's life easier.

GitHub Account
--------------

If you don't have a GitHub account please [register](http://github.com/join) your account

Fork the repo
--------------

First, you'll need to fork this repo https://github.com/buildtesters/buildtest-cori

You might need to setup your SSH keys in your git profile if you are using ssh option for cloning. For more details on
setting up SSH keys in your profile, follow instruction found in
https://help.github.com/articles/connecting-to-github-with-ssh/

SSH key will help you pull and push to repository without requesting for password for every commit. Once you have forked the repo, clone your local repo

```
git clone git@github.com:YOUR\_GITHUB\_LOGIN/buildtest-cori.git
```

Adding Upstream Remote
-----------------------

First you need to add the ``upstream`` repo, to do this you can issue the
following:

```
git remote add upstream git@github.com/buildtesters/buildtest-cori.git
```

The ``upstream`` tag is used to sync your local fork with upstream repo.

Make sure you set your user name and email set properly in git configuration.
We don't want commits from unknown users. This can be done by setting the following:

```
git config user.name "First Last"
git config user.email "abc@example.com"
```

For more details see [First Time Git Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)

Sync your branch from upstream
-------------------------------

The ``devel`` from upstream will get Pull Requests from other contributors, in-order
to sync your forked repo with upstream, run the commands below::

```
cd buildtest-cori
git checkout devel
git fetch upstream devel
git pull upstream devel
```

Once the changes are pulled locally you can sync your upstream fork as follows:

```
git checkout devel
git push origin devel
```

Repeat this same operation with ``master`` branch if you want to sync with
upstream repo

Feature Branch
------------------

Please make sure to create a new branch when adding and new feature. Do not
push to ``master`` or ``devel`` branch on your fork or upstream.

Create a new branch from ``devel`` as follows:

```
 cd buildtest-cori
 git checkout devel
 git checkout -b featureX
```

Once you are ready to push to your fork repo do the following:

```
git push origin featureX
```

Once the branch is created in your fork, you can issue a Pull Request to ``devel``
branch for ``upstream`` repo (https://github.com/buildtesters/buildtest-cori)

General Tips
-------------

1. It's good practice to link PR to an issue during commit message. Such as
stating ``Fix #132`` for fixing issue 132.

2. If you have an issue, ask your question in slack before reporting issue. If
your issue is not resolved check any open issues for resolution before creating
a new issue.

3. There should not be any branches other than ``master`` or ``devel``. Feature
branches should be pushed to your fork and not upstream repo. 

Contributing Tests
-------------------

All incoming PRs for tests are pushed to ``devel`` branch which uses HEAD of devel branch from buildtest repo (https://github.com/buildtesters/buildtest).
The ``master`` branch for buildtest-cori is using latest version of buildtest, whenever there is a new release on buildtest (https://github.com/buildtesters/buildtest/releases) the 
``master`` branch for buildtest-cori can be synced with latest version. 

Please make sure you sync your buildtest with `devel` when you contribute tests and buildspec. Please refer to buildtest [contributing guide](https://buildtest.readthedocs.io/en/devel/contributing.html) for more details.


Application tests are stored in [apps](https://github.com/buildtesters/buildtest-cori/tree/devel/apps) directory. Tests are categorized by application, please consider adding test in one of the 
appropriate directories or create a new directory. Tests from [E4S Testsuite](https://github.com/E4S-Project/testsuite) are located in [e4s](https://github.com/buildtesters/buildtest-cori/tree/devel/e4s) directory. This should be used for 
testing E4S stack on Cori. 

If you are adding a buildspec in your PR, please add a test description using ``description`` field. Please add test with appropriate tagname using ``tags`` field. For instance all
e4s tests are set to ``e4s`` tag name. This allows all e4s tests to be run via ``buildtest build --tags e4s``. 

Please add yourself to ``maintainers`` field which helps contact individual when test fails. When you contribute your test and buildspec please test this locally. The buildspec must be 
valid, this can be done by building the test via ``buildtest build -b /path/to/test.yml``. 


Resolving PR Merge Conflicts
-----------------------------

Often times, you may start a feature branch and your PR get's out of sync with
``devel`` branch which may lead to conflicts, this is a result of merging incoming
PRs that may cause upstream `HEAD` to change over time which can cause merge conflicts.
This may be confusing at first, but don't worry we are here to help. For more details
about merge conflicts click [here](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-merge-conflicts).

Syncing your feature branch with devel is out of scope for this documentation,
however you can use the steps below as a *guide* when you run into this issue.

You may want to take the steps to first sync devel branch and then
selectively rebase or merge ``devel`` into your feature branch.

First go to ``devel`` branch and fetch changes from upstream:

```
git checkout devel
git fetch upstream devel
```

Note you shouldn't be making any changes to your local ``devel`` branch, if
``git fetch`` was successful you can merge your ``devel`` with upstream as follows:

```
 git merge upstream/devel
```

Next, navigate to your feature branch and sync feature changes with devel::

```
git checkout <feature-branch>
git merge devel
```

!!! note

   Running above command will sync your feature branch with ``devel`` but you may have some file conflicts depending on files changed during PR. You will need to resolve them manually before pushing your changes

Instead of merge from ``devel`` you can rebase your commits interactively when syncing with ``devel``. This can be done by running:

```
git rebase -i devel
```

Once you have synced your branch push your changes and check if file conflicts are resolved in your Pull Request:

```
 git push origin <feature-branch>
 ```
