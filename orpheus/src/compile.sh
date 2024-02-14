
gcc -fopenmp -fPIC -Wall -c assign.c -std=c99
gcc -fopenmp -fPIC -Wall -c discrete.c -std=c99
gcc -fopenmp -fPIC -Wall -c spatialhash.c -std=c99

gcc -fopenmp -fPIC -Wall -shared -o ../../orpheus_clib.so assign.c discrete.c spatialhash.c


