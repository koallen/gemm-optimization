CC=gcc
CFLAGS=-O3 -fomit-frame-pointer -Wall -DNDEBUG
BLASDIR=/home/koallen/blis

.PHONY: test clean step0 step1

main: driver.o my_dgemm.o after_step.o
	$(CC) -o $@ $^ $(BLASDIR)/lib/*.a -lpthread -lm

test: main
	./main

step0:
	cp $@/my_dgemm.c .
	cp $@/after_step.c .
	make test

step1:
	cp $@/my_dgemm.c .
	cp $@/after_step.c .
	make test

clean:
	rm -f main *.o
