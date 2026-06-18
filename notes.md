I want the README.md to read:
1. Overview
2. Quickstart
2.1. Docker build + compilation
2.2. Run scripts to generate some result from the paper
2.3. Create your own scene and run it
3. Results
4. Changes to the code
5. Citation

----
docker compose build
docker compose up -d
docker exec -it bash onemorevertexneehierarchicalgeometrysampling-mitsuba-1
cp build/config-linux-gcc.py config.py
python2.7 $(which scons)
source setpath.sh
mitsuba scenes/experiments/1_wildcard_spotlight/wildcard_spotlight_quads.xml

ln -s /home/mitsuba/data /home/mitsuba/ppg/mitsuba/data
cd ppg/mitsuba
cp build/config-linux-gcc.py config.py
python2.7 $(which scons)


scenes:
- basic:
    - basic: old because additional_vertex name
    - bvhintegrators_spotlight: 
    - changing_bsdf_albedo
    - changing_bsdf_roughness
    - time_complexity_increasing_quads
    - veach
- principled_bsdf: visual test to see that the principled bsdf implemented is correct
- experiments:


