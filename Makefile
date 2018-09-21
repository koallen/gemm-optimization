CC=gcc
CFLAGS=-O3 -Wall -DNDEBUG

all: bin openblas

.PHONY:
bin:
	mkdir -p bin

openblas: driver.o
	$(CC) -o bin/$@ $< -lopenblas

test:
	./bin/openblas

.PHONY:
clean:
	rm -r bin *.o
