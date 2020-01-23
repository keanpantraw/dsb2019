#!/bin/bash
docker exec -it lab jupyter nbconvert --to script "/code/dsb2019/notebooks/$1"
