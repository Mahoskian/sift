BUILD_DIR := build

.PHONY: all build configure run viz clean help

all: build

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  configure   Configure the CMake build directory (first-time setup)"
	@echo "  build       Compile the sift binary  [default]"
	@echo "  run         Run the binary directly, e.g.: make run ARGS=\"hash ./input\""
	@echo "  viz         Launch the interactive visualizer"
	@echo "  clean       Remove the build directory"

configure:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release $(if $(shell which ninja),-G Ninja,)

build:
	cmake --build $(BUILD_DIR) --parallel

run: build
	./$(BUILD_DIR)/sift $(ARGS)

viz:
	cd viz && uv run visualize.py

clean:
	rm -rf $(BUILD_DIR)
