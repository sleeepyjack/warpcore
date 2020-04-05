INCDIRS := ../include

CC := g++
NVCC := nvcc
STD := c++14
OPT := 3
CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp -DNDEBUG
XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
NVCCGENCODE = -gencode arch=compute_70,code=sm_70
NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda

INCS := $(foreach dir, $(INCDIRS), $(wildcard $(dir)/*.cuh $(dir)/*.h $(dir)/*.hpp))
INCPARAMS := $(addprefix -I, $(INCDIRS))

all: basic_usage_from_host advanced_usage_from_device multi_value_hash_table bucket_list_hash_table

basic_usage_from_host: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/basic_usage_from_host.cu -o bin/basic_usage_from_host.out

advanced_usage_from_device: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/advanced_usage_from_device.cu -o bin/advanced_usage_from_device.out

multi_value_hash_table: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/multi_value_hash_table.cu -o bin/multi_value_hash_table.out

bucket_list_hash_table: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/bucket_list_hash_table.cu -o bin/bucket_list_hash_table.out

counting_hash_table: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) src/counting_hash_table.cu -o bin/counting_hash_table.out

unique_random_generator: ${INCS} | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) -I../util/ -I../ext/kiss_rng/include/ src/unique_random_generator.cu -o bin/unique_random_generator.out

debug: OPT := 0
debug: CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
debug: XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
debug: NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda -g -Xptxas -v -UNDEBUG -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -DNDEBUG
profile: all

clean:
	$(RM) -r bin

bin:
	mkdir -p $@

.PHONY: clean all bin
