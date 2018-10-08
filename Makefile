CC=gcc
CFLAGS=-O3 -march=native -Wall -DNDEBUG

.PHONY: test clean step0

main: driver.o my_dgemm.o
	$(CC) -o main $^ -lopenblas

test: main
	./main

step0:
	cp step0/my_dgemm.c .
	make test

clean:
	rm -f main *.o
