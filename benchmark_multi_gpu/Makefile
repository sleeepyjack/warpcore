SRCDIR := src
BINDIR := bin
INCDIR := ../include

INCDIRS := ../include ../ext/gossip/include

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -Wall  -Wextra -fopenmp -DNDEBUG
NVCCGENCODE = \
              -gencode arch=compute_70,code=sm_70
            #   -gencode arch=compute_61,code=sm_61
NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) --expt-extended-lambda -DNDEBUG -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

INCS := $(foreach dir, $(INCDIRS), $(wildcard $(dir)/*.cuh $(dir)/*.h $(dir)/*.hpp))
INCPARAMS := $(addprefix -I, $(INCDIRS))

all: single_value

single_value: ${INCS} $(BINDIR)/plan_parser.o | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) $(BINDIR)/plan_parser.o $(SRCDIR)/single_value_benchmark.cu -o $(BINDIR)/single_value_benchmark.out

$(BINDIR)/plan_parser.o: ${INCS} | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) -c ../ext/gossip/include/plan_parser.cpp -o $(BINDIR)/plan_parser.o

debug: OPT := 0
debug: CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
debug: XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
debug: NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda -g -Xptxas -v -UNDEBUG -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v -DNDEBUG
profile: all

clean:
	$(RM) -r $(BINDIR)

$(BINDIR):
	mkdir -p $@

.PHONY: clean all $(BINDIR)
