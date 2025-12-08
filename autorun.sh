#!/bin/bash

cd /home/yoojk20/workspace/mnpbem_simulation

./master.sh --structure ./config/structure/sphere_test/config_30nm.py --simulation ./config/simulation/sphere_test/config_30nm.py --verbose &
./master.sh --structure ./config/structure/sphere_test/config_50nm.py --simulation ./config/simulation/sphere_test/config_50nm.py --verbose &
./master.sh --structure ./config/structure/sphere_test/config_70nm.py --simulation ./config/simulation/sphere_test/config_70nm.py --verbose &
./master.sh --structure ./config/structure/sphere_test/config_90nm.py --simulation ./config/simulation/sphere_test/config_90nm.py --verbose

echo "JOB DONE"
