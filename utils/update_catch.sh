#!/bin/sh
wget https://raw.githubusercontent.com/catchorg/Catch2/master/single_include/catch2/catch.hpp -O "$(dirname "$0")/../tests/include/catch.hpp"
