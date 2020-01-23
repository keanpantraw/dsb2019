#!/bin/bash
docker exec -it lab jupyter nbconvert --to notebook --execute "/code/dsb2019/notebooks/$1" --to html --ExecutePreprocessor.timeout=-1
