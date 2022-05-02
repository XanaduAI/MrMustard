ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: help
help:
	@echo "  format [check=1]"

.PHONY: format
format:
ifdef check
	./bin/format --check --cfversion 13 ./src ./tests
else
	./bin/format --cfversion 13 ./src ./tests
endif

.PHONY: test
test:
	rm -rf ./BuildTests
	cmake . -BBuildTests -DCMAKE_BUILD_TYPE=Debug
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test
