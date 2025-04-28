INCD_PATH = -I /usr/include/opencv4 $(shell pkg-config --cflags --libs opencv4)
LINK_FLAGS = $(shell pkg-config --libs opencv4)

CC := g++
FLAGS := 

TARGET := test
MAIN_FILE := cam_feed.cpp
DEPS := 

default:
	$(CC) $(MAIN_FILE) -o $(TARGET) $(FLAGS) $(INCD_PATH)


.PHONY: clean
clean:
	$(RM) TARGET
	$(MAKE)
  