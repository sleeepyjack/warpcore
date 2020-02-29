SRCDIR := src
BINDIR := bin
INCDIR := ../include

INCDIRS := ../include

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -Wall  -Wextra -fopenmp -DNDEBUG
NVCCGENCODE = -gencode arch=compute_70,code=sm_70
NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) --expt-extended-lambda -DNDEBUG -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

INCS := $(foreach dir, $(INCDIRS), $(wildcard $(dir)/*.cuh $(dir)/*.h $(dir)/*.hpp))
INCPARAMS := $(addprefix -I, $(INCDIRS))

all: single_value multi_value counting bloom_filter

single_value: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/single_value_benchmark.cu -o bin/single_value_benchmark.out

multi_value: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/multi_value_benchmark.cu -o bin/multi_value_benchmark.out

counting: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/counting_benchmark.cu -o bin/counting_benchmark.out

bloom_filter: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/bloom_filter_benchmark.cu -o bin/bloom_filter_benchmark.out

debug: OPT := 0
debug: CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
debug: XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
debug: NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda -g -Xptxas -v -UNDEBUG -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v -DNDEBUG
profile: all

clean:
	$(RM) -r bin

$(BINDIR):
	mkdir -p $@

.PHONY: clean all bin
