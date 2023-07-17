## Gitlab Runner

This project will run CI jobs using [collaboration account](https://docs.nersc.gov/accounts/collaboration_accounts/) `bdtest`. You can login to this user via laptop. We recommend using `sshproxy` so you can avoid typing password 
for every ssh connection.

This account has a gitlab runner installed on several login nodes. If you want to upgrade the gitlab runner to new version, please see [Install Gitlab Runner](https://docs.gitlab.com/runner/install/) for more details.

The `gitlab-runner` should be accessible when you login, if you need to upgrade to new version, please put this in the same path

```console
bdtest@perlmutter:login07:~/bin> which gitlab-runner
/global/homes/b/bdtest/bin/gitlab-runner
```

The gitlab runner configuration is located in `$HOME/.gitlab-runner` directory, shown below are the runner configuration files on Perlmutter and Muller. Note we have
runners on different login nodes.

```console
bdtest@perlmutter:login07:~/bin> ls -l ~/.gitlab-runner/*.config.toml
-rw------- 1 bdtest bdtest 1119 Jul 11 10:01 /global/homes/b/bdtest/.gitlab-runner/muller-login01.config.toml
-rw------- 1 bdtest bdtest 1123 Jul 11 09:55 /global/homes/b/bdtest/.gitlab-runner/perlmutter-login07.config.toml
-rw------- 1 bdtest bdtest 1123 Jul 11 09:56 /global/homes/b/bdtest/.gitlab-runner/perlmutter-login08.config.toml
```

Once you are logged in, you can check status of the gitlab runner using **systemctl**. For instance to check status of runner on Perlmutter you can run

```console
bdtest@perlmutter:login07:> systemctl --user status perlmutter-login07
● perlmutter-login07.service - Gitlab runner for bdtest on perlmutter
     Loaded: loaded (/global/homes/b/bdtest/.config/systemd/user/perlmutter-login07.service; enabled; vendor preset: disabled)
     Active: active (running) since Mon 2023-07-17 10:07:38 PDT; 3h 10min ago
   Main PID: 1027247 (gitlab-runner)
      Tasks: 47 (limit: 353894)
     Memory: 20.8M
        CPU: 6.016s
     CGroup: /user.slice/user-99914.slice/user@99914.service/app.slice/perlmutter-login07.service
             └─ 1027247 /global/homes/b/bdtest/bin/gitlab-runner run -c /global/homes/b/bdtest/.gitlab-runner/perlmutter-login07.config.toml
```

The systemd service configuration are located in directory ``$HOME/.config/systemd/user``, shown below are the systemd service files (*.service).

```console
bdtest@perlmutter:login07:~/bin> ls -l $HOME/.config/systemd/user/*.service
-rw-rw---- 1 bdtest bdtest 326 Jul 11 10:02 /global/homes/b/bdtest/.config/systemd/user/muller-login01.service
-rw-rw---- 1 bdtest bdtest 330 Jul 11 09:53 /global/homes/b/bdtest/.config/systemd/user/perlmutter-login07.service
-rw-rw---- 1 bdtest bdtest 330 Jul 11 09:56 /global/homes/b/bdtest/.config/systemd/user/perlmutter-login08.service
```

If you want to start/stop/restart the service, please login to the appropriate node and do the following. 
Shown below are changes to runner `perlmutter-login07` performed on `login07`:

```
# restart service
systemctl --user restart perlmutter-login07

# stop service
systemctl --user stop perlmutter-login07

# start service 
systemctl --user start perlmutter-login07
```

## Register a Runner

If you want to register a new runner, please use the `register.sh` script and go to the appropriate login node. The script is the following

```bash
bdtest@perlmutter:login07:~> cat register.sh
#!/bin/bash

read -sp "Registration Token?" TOKEN
gitlab-runner register \
	--url https://software.nersc.gov \
	--registration-token ${TOKEN} \
       	--tag-list ${NERSC_HOST}-${USER} \
	--name "bdtest Runner on ${NERSC_HOST} at login: ${HOSTNAME}" \
	--executor custom \
	--custom-config-exec "/global/homes/b/bdtest/jacamar/binaries/jacamar-auth" \
	--custom-config-args "-u" \
	--custom-config-args "config" \
        --custom-config-args "--configuration" \
	--custom-config-args "/global/homes/b/bdtest/.gitlab-runner/jacamar.toml" \
	--custom-prepare-exec "/global/homes/b/bdtest/jacamar/binaries/jacamar-auth" \
	--custom-prepare-args "prepare" \
	--custom-run-exec  "/global/homes/b/bdtest/jacamar/binaries/jacamar-auth" \
	--custom-run-args "run" \
	--custom-cleanup-exec "/global/homes/b/bdtest/jacamar/binaries/jacamar-auth" \
	--custom-cleanup-args "cleanup" \
        --custom-cleanup-args "--configuration" \
        --custom-cleanup-args "/global/homes/b/bdtest/.gitlab-runner/jacamar.toml" \
	--config "/global/homes/b/bdtest/.gitlab-runner/${NERSC_HOST}-${HOSTNAME}.config.toml"
```

The script will request a runner token that can be obtained at [Settings > CI/CD > Runners](https://software.nersc.gov/NERSC/buildtest-nersc/-/settings/ci_cd). Please copy 
the token when prompted and paste it in the terminal.

This script will create a gitlab runner configuration for Perlmutter in `$HOME/.gitlab-runner` directory. Once you 
have registered the runner, you can create a new systemd service file in `$HOME/.config/systemd/user` directory. 
Please copy one of the service file and make the necessary change. 

## Jacamar

[Jacamar CI](https://gitlab.com/ecp-ci/jacamar-ci) is a HPC focused CI/CD driver for gitlab custom executor that allows us to run jobs on NERSC systems easily. The jacamar 
binaries are available in `$HOME/jacamar` denoted by each version to allow us to switch between versions easily. The binaries are symlinked to `$HOME/jacamar/binaries` to allow 
a single place to reference the binary for gitlab runner configuration.

```console
bdtest@perlmutter:login07:~> ls -l ~/jacamar/
total 2
drwxrwx--- 2 bdtest bdtest 512 Jul 11 07:43 binaries
drwxrwx--- 2 bdtest bdtest 512 Jul 11 07:42 v0.15.0

bdtest@perlmutter:login07:~> ls -l ~/jacamar/binaries/
total 2
lrwxrwxrwx 1 bdtest bdtest 18 Jul 11 07:43 jacamar -> ../v0.15.0/jacamar
lrwxrwxrwx 1 bdtest bdtest 23 Jul 11 07:43 jacamar-auth -> ../v0.15.0/jacamar-auth
```

If you want to change the jacamar version, please install the latest version and create a directory naming the version and store the files in their. Next, update the symbolic link
to the latest version.

The jacamar configuration can be found at `~/.gitlab-runner/jacamar.toml`
