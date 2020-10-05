INCDIRS := ../include

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -Wall  -Wextra -fopenmp -DNDEBUG
NVCCGENCODE = -gencode arch=compute_70,code=sm_70
NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) --expt-extended-lambda --expt-relaxed-constexpr -DNDEBUG -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

INCS := $(foreach dir, $(INCDIRS), $(wildcard $(dir)/*.cuh $(dir)/*.h $(dir)/*.hpp))
INCPARAMS := $(addprefix -I, $(INCDIRS))

all: bin/single_value_benchmark.out \
     bin/multi_value_benchmark.out \
     bin/multi_bucket_benchmark.out \
     bin/counting_benchmark.out \
     bin/bloom_filter_benchmark.out \
     bin/bucket_list_benchmark.out

bin/single_value_benchmark.out: src/single_value_benchmark.cu src/common.cuh ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/single_value_benchmark.cu -o bin/single_value_benchmark.out

bin/multi_value_benchmark.out: src/multi_value_benchmark.cu src/common.cuh ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/multi_value_benchmark.cu -o bin/multi_value_benchmark.out

bin/multi_bucket_benchmark.out: src/multi_bucket_benchmark.cu src/common.cuh ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/multi_bucket_benchmark.cu -o bin/multi_bucket_benchmark.out

bin/counting_benchmark.out: src/counting_benchmark.cu src/common.cuh ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/counting_benchmark.cu -o bin/counting_benchmark.out

bin/bloom_filter_benchmark.out: src/bloom_filter_benchmark.cu src/common.cuh ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/bloom_filter_benchmark.cu -o bin/bloom_filter_benchmark.out

bin/bucket_list_benchmark.out: src/bucket_list_benchmark.cu src/common.cuh ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/bucket_list_benchmark.cu -o bin/bucket_list_benchmark.out

debug: OPT := 0
debug: CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
debug: XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
debug: NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda -g -Xptxas -v -UNDEBUG -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v -DNDEBUG
profile: all

clean:
	$(RM) -r bin

bin:
	mkdir -p $@

.PHONY: clean all debug profile
