CC=gcc
CCFLAGS=-Wall -Wextra
LDFLAGS = -lm
IN=exmpl.c
OUT=ml
LIB=ml.h
SRC=src
BUILD=build

fresh:
	mkdir -p $(BUILD)
	mkdir -p $(SRC)
	touch $(SRC)/$(IN) && touch $(SRC)/$(LIB)
start:
	mkdir -p $(BUILD)
	mkdir -p $(SRC)
	mv $(IN) $(SRC) && mv $(LIB) $(SRC)
all:
	$(CC) -o $(BUILD)/$(OUT) $(SRC)/$(IN) $(SRC)/$(LIB) $(CCFLAGS) $(LDFLAGS)
run:
	./$(BUILD)/$(OUT)
re:
	make all
	clear
	make run
clean:
	mv $(SRC)/* .
	rm -rf $(BUILD) && rm -rf $(SRC)
wipe:
	rm -rf $(BUILD) && rm -rf $(SRC)
