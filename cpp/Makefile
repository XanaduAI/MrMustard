.PHONY: help
help:
	@echo "  format [check=1]"

.PHONY: format
format:
ifdef check
	./bin/format --check --cfversion $(if $(version:-=),$(version),0) ./src ./tests
else
	./bin/format --cfversion $(if $(version:-=),$(version),0) ./src ./tests
endif
