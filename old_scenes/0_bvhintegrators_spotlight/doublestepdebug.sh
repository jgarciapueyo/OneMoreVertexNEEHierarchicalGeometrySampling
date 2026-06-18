# run ./build/release/mitsuba/mitsuba scenes/basic/bvhintegrators/doublestep.xml -o scenes/basic/bvhintegrators/doublestep_neemult_1000.exr -Dd1=1 -Dd2=0 -Dd3=0 -Dd4=0
# and all other combinations of d1, d2, d3, d4 in {0, 1} to get the 16 images for debugging the two-step method with/without NEE and with/without BVH pdf. The images will be saved in the same folder as the XML file.


for d1 in 0 1; do
    for d2 in 0 1; do
        for d3 in 0 1; do
            for d4 in 0 1; do
                output="scenes/basic/bvhintegrators/doublestep_spherical_l4_${d1}${d2}${d3}${d4}.exr"
                ./build/release/mitsuba/mitsuba scenes/basic/bvhintegrators/doublestep.xml -o $output -Dd1=$d1 -Dd2=$d2 -Dd3=$d3 -Dd4=$d4
            done
        done
    done
done

