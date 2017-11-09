CC = nvcc
CFLAGS = -g
LIBS = -lopencv_core -lopencv_highgui
TARGET = edge_detector

default: $(TARGET)

all: $(TARGET)

$(TARGET): src/main.cu
	$(CC) $(CFLAGS) -o $(TARGET) src/main.cu $(LIBS)

clean:
	$(RM) $(TARGET)
