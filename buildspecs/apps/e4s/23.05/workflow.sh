#!/bin/bash
rm -rf $HOME/spack-workspace/perlmutter/modules
rm -rf $HOME/spack-workspace/perlmutter/software
module load e4s/23.05
source $SPACK_ROOT/bin/spack-setup.sh
spack env status
spack spec zlib
rm -rf $HOME/e4s-demo
spack env create -d $HOME/e4s-demo spack.yaml
spack env activate -d $HOME/e4s-demo
ls -l $HOME/e4s-demo
spack config list
echo  $SPACK_SYSTEM_CONFIG_PATH
ls -l $SPACK_SYSTEM_CONFIG_PATH
cat $SPACK_ROOT/etc/spack/config.yaml
spack clean -m
spack install
spack location -i papi
cat $SPACK_ROOT/etc/spack/modules.yaml
spack module tcl refresh --delete-tree -y
spack module tcl find --full-path papi
module use $HOME/spack-workspace/perlmutter/modules/$(spack arch)
ml -t av papi
