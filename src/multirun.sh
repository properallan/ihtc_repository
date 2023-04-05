#!/bin/bash 

export DISPLAY=:0.0

declare -i j=0
declare -i l=3
for py_file in `find ./multiple_run -maxdepth 1 -name '*.py'`; 
do  
    j=$j+1
    #cd $py_file | cut -d "\\" -f 1
    #echo $py_file | cut -d "\\" -f 1
    #python $py_file -f True done   
    
    #echo $py_file 
    #echo $py_file | cut -d "/" -f3
    tmp_dir=$( echo $py_file | cut -d "." -f1,2)
    #mkdir -p $tmp_dir"/tmp_$j"
    mkdir -p $tmp_dir
    cp $py_file $tmp_dir
    cd $tmp_dir
    #xterm -hold -e echo $(pwd) &
    #xterm -hold -e python 
    #echo $tmp_dir/$(echo $py_file | cut -d "/" -f3)
    xterm -hold -e python $(echo $py_file | cut -d "/" -f3) -f True& 
    cd ../..

    
done

#while [[ $j < $l ]]; do
#    echo 'run ' $j

#    mkdir -p "./tmp_$j"
#    cd "./tmp_$j"
#    xterm -hold -e python ../run_python_multi.py &
    #xterm -hold -e echo $(pwd) &
#    cd ..
#    j=$j+1
#done

