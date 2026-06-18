cp build/config-linux-gcc.py config.py
python2.7 $(which scons) -j$(nproc) $@
# To execute mitsuba, source setpath.sh should be used first.