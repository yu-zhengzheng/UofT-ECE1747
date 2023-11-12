# Program Name

1747 assignment 1

## Prerequisites

Before running the program, ensure you have the following installed:
- [MPI (Message Passing Interface)](https://www.open-mpi.org/)
- [C++ Compiler](#) 

## Running the Program

### Mode 1

To run the program in Mode 1, use the following command:

```bash
mpic++ -o main main.cpp; ./main 1 a2 a3
```

### Mode 2

To run the program in Mode 2, use the following command:

```bash
mpic++ -o main main.cpp; ./main 1 a2 a3
```
### Mode 3

To run the program in Mode 3, use the following command:

```bash
mpic++ -o main main.cpp; mpirun -np a1 ./main 3 a2 a3
```
### Arguments

a1: the desired number of processes.

a2: the desired number of threads/worker threads per leader process.

a3: the desired number of particles to be computed.