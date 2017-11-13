CC = nvcc
GPP = g++
CFLAGS = -g
LIBS = -lopencv_core -lopencv_highgui

SERIAL = serial
PARALLEL = parallel

SERIAL_TARGET = edge_detector_serial
PARALLEL_TARGET = edge_detector

default: $(PARALLEL)

all: $(PARALLEL) $(SERIAL) 

$(SERIAL): $(SERIAL_TARGET)

$(PARALLEL): $(PARALLEL_TARGET)

$(PARALLEL_TARGET): src/main.cu
	$(CC) $(CFLAGS) -o $(PARALLEL_TARGET) src/main.cu $(LIBS)

$(SERIAL_TARGET): src/serial.cpp
	$(GPP) $(CFLAGS) -o $(SERIAL_TARGET) src/serial.cpp $(LIBS)

clean:
	$(RM) $(PARALLEL_TARGET)
	$(RM) $(SERIAL_TARGET)
