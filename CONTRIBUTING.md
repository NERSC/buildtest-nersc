Getting Started
================

Contribution is not easy, so we created this document to describe how to get you setup
so you can contribute back and make everyone's life easier. All contributions will be made via
[merge request](https://software.nersc.gov/NERSC/buildtest-cori/-/merge_requests).
 We have disabled push to **devel** branch, therefore all changes must go through merge request.

First, you'll need to fork this repo https://software.nersc.gov/NERSC/buildtest-cori/ in your userspace and clone 
the repo.


Adding Upstream Remote
-----------------------

First you need to add the `upstream` repo, to do this by running the following:

```
git remote add upstream https://software.nersc.gov/NERSC/buildtest-cori.git
```

The `upstream` tag is used to sync your local fork with upstream repo. If your remotes are setup properly you should have something like this:

```
siddiq90@cori01> git remote -v
origin	https://software.nersc.gov/siddiq90/buildtest-cori (fetch)
origin	https://software.nersc.gov/siddiq90/buildtest-cori (push)
upstream	https://software.nersc.gov/NERSC/buildtest-cori.git (fetch)
upstream	https://software.nersc.gov/NERSC/buildtest-cori.git (push)
```

Make sure you set your user name and email set properly in git configuration.
We don't want commits from unknown users. This can be done by setting the following:

```
git config user.name "First Last"
git config user.email "abc@example.com"
```

This will save the configuration local to this repo, Alternately you can setup your *user.name* and *user.email* 
in global setting so its configured for all projects which can be done by running the following:

```
git config --global user.name "First Last"
git config --global user.email "abc@example.com"
```

The global git configuration is stored in `~/.gitconfig`. If you have done this correctly you may have something like as follows

```
siddiq90@cori01> cat ~/.gitconfig
[user]
	name = Shahzeb Siddiqui
	email = shahzebmsiddiqui@gmail.com
```

For more details see [First Time Git Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)

Sync your branch from upstream
-------------------------------

The `devel` from upstream will get Pull Requests from other contributors, in-order
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


Feature Branch
------------------

Please make sure to create a new branch when adding and new feature. Do not
push to `devel` branch on your fork or upstream.

Create a new branch from `devel` as follows:

```
 cd buildtest-cori
 git checkout devel
 git checkout -b featureX
```

Once you are ready to push to your fork repo do the following:

```
git push origin featureX
```

Once the branch is created in your fork, you can issue a Merge Request to `devel`
branch for upstream repo (https://software.nersc.gov/siddiq90/buildtest-cori.git).

Default Branch
---------------

The default branch is `devel`. This branch is [protected branch](https://docs.gitlab.com/ee/user/project/protected_branches.html) 
which prevents users from accidently deleting the branch. 


Contributing Tests
-------------------

All incoming PRs for tests are pushed to `devel` branch which uses HEAD of devel branch from buildtest repo (https://github.com/buildtesters/buildtest).

Please make sure you sync your buildtest with `devel` when you contribute tests and buildspec. Please refer to buildtest [contributing guide](https://buildtest.readthedocs.io/en/devel/contributing.html) for more details.

Application tests are stored in [apps](https://software.nersc.gov/NERSC/buildtest-cori/-/tree/devel/buildspecs/apps) directory. Tests are categorized by application, please consider adding test in one of the 
appropriate directories or create a new directory. Tests from [E4S Testsuite](https://github.com/E4S-Project/testsuite) are located in [e4s](https://software.nersc.gov/NERSC/buildtest-cori/-/tree/devel/buildspecs/e4s) directory. This should be used for testing E4S stack on Cori. 

If you are adding a buildspec in your PR, please add a test description using `description` field. Please add test with appropriate tagname using `tags` field. For instance all e4s tests are set to `e4s` tag name. This allows all e4s tests to be run via `buildtest build --tags e4s`. 

Please add yourself to `maintainers` field which helps contact individual when test fails. When you contribute your test and buildspec please test this locally. 

Resolving PR Merge Conflicts
-----------------------------

Often times, you may start a feature branch and your PR get's out of sync with
`devel` branch which may lead to conflicts, this is a result of merging incoming
PRs that may cause upstream `HEAD` to change over time which can cause merge conflicts.
This may be confusing at first, but don't worry we are here to help. For more details
about merge conflicts click [here](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-merge-conflicts).

Syncing your feature branch with devel is out of scope for this documentation,
however you can use the steps below as a *guide* when you run into this issue.

You may want to take the steps to first sync devel branch and then
selectively rebase or merge `devel` into your feature branch.

First go to `devel` branch and fetch changes from upstream:

```
git checkout devel
git fetch upstream devel
```

Note you shouldn't be making any changes to your local `devel` branch, if `git fetch` was successful you can merge your `devel` with upstream as follows:

```
 git merge upstream/devel
```

Next, navigate to your feature branch and sync feature changes with devel::

```
git checkout <feature-branch>
git merge devel
```

!!! note:

   Running above command will sync your feature branch with `devel` but you may have some file conflicts depending on files changed during PR.
   You will need to resolve them manually before pushing your changes

Instead of merge from `devel` you can rebase your commits interactively when syncing with `devel`. This can be done by running:

```
git rebase -i devel
```

Once you have synced your branch push your changes and check if file conflicts are resolved in your Pull Request:

```
git push origin <feature-branch>
```

Project Maintainers
---------------------

If you need elevated priviledge to project settings please contact the maintainer

- Shahzeb Siddiqui, [@shahzebsiddiqui](http://github.com/shahzebsiddiqui)
