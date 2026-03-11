#!/bin/bash
source /root/exchange/install/setup.bash
export PYTHONPATH=/home/semanticnuc/Desktop/TiagoSara/Final/lost-3dsg/install/lost3dsg/local/lib/python3.10/dist-packages:$PYTHONPATH
export PYTHONPATH=/root/exchange/perception_module:$PYTHONPATH
export PYTHONPATH=/root/exchange/perception_module/lost3dsg/perception_module:$PYTHONPATH
export PYTHONPATH=/root/exchange/perception_module/efficientvit:$PYTHONPATH
python3 -c "import lost3dsg; print('lost3dsg from:', lost3dsg.__file__)"
python3 -c "from lost3dsg.msg import ObjectDescriptionArray; print('MSG OK')"
ros2 run lost3dsg perception
