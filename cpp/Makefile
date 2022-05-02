ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: help
help:
	@echo "  format [check=1]"
	@echo "  test"

.PHONY: format
format:
ifdef check
	./bin/format --check --cfversion 13 ./src ./tests ./examples
else
	./bin/format --cfversion 13 ./src ./tests ./examples
endif

.PHONY: test examples
test:
	rm -rf ./BuildTests
	cmake . -BBuildTests -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test

examples:
	rm -rf ./BuildExamples
	cmake . -BBuildExamples -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-flto" -DENABLE_NATIVE=ON -DENABLE_OPENMP=ON
	cmake --build ./BuildExamples
