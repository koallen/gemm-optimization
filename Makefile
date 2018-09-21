CC=gcc
CFLAGS=-O3 -Wall

all: bin openblas

.PHONY:
bin:
	mkdir -p bin

openblas: driver.c
	$(CC) -o bin/$@ $< -lopenblas

test:
	./bin/openblas
