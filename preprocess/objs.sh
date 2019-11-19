#!/usr/bin/env bash
cd /media/z/Data/Object_Searching/code/Environment/houses/
houselist=*
echo $houselist

dataset=/media/z/Data/suncg/house/

for houseid in $houselist
do
    if [ "$houseid" -eq "real" ]
    then
        echo ${dataset}${houseid}
    else
        cd ${dataset}${houseid}
        echo $(pwd)
        ls
        # /media/z/Data/Object_Searching/code/Environment/House3D/SUNCGtoolbox/gaps/bin/x86_64/scn2scn house.json house.obj
    fi
done