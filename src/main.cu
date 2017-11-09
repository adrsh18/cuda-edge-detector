#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void img_kernel(unsigned char *d_inputImage, unsigned char *d_outputImage, int rows, int cols) {
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (y < cols && x < rows) {
        int idx = x * cols + y;
        d_outputImage[idx] = d_inputImage[idx];
    }
    return;
}

__global__ void nms_kernel(unsigned char *d_imageGradient, unsigned char *d_gradientAngle, unsigned char *d_outputImage, int rows, int cols) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = r*cols+c, neighborOne = idx, neighborTwo = idx;
    if (r >= rows || c >= cols) {
        return;
    }

    if (r < 1 || r >= rows-1 || c < 1 || c >= cols-1) {
        d_outputImage[idx] = (unsigned char) 0;
    } else {
        switch(d_gradientAngle[idx]) {
            case 0:
                neighborOne = idx - 1; neighborTwo = idx + 1;
                break;
            
            case 45:
                neighborOne = (r-1)*cols+c-1; neighborTwo = (r+1)*cols+c+1;
                break;

            case 90:
                neighborOne = idx - cols; neighborTwo = idx + cols;
                break;

            default:
                neighborOne = (r-1)*cols+c+1; neighborTwo = (r+1)*cols+c-1;
                break;

        }
        if (d_imageGradient[idx] > d_imageGradient[neighborOne] && d_imageGradient[idx] > d_imageGradient[neighborTwo]) {
            d_outputImage[idx] = d_imageGradient[idx];
        } else {
            d_outputImage[idx] = (unsigned char) 0;
        }
    }
}

__global__ void sobel_kernel(unsigned char *d_filteredImage, unsigned char *d_imageGradient, unsigned char *d_gradientAngle, int rows, int cols, int threshold) {
    int x_filter[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int y_filter[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    float gradient = 0.0, gx = 0.0, gy = 0.0, angle = 0.0;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    int mid = 1, idx = r*cols+c;

    if (r >= rows || c >= cols) {
         return;
    }

    if (r < mid || r >= rows-mid || c < mid || c >= cols-mid) {
        d_imageGradient[idx] = (unsigned char) 0;
        d_gradientAngle[idx] = (unsigned char) 0;
    } else {
        for (int i = -mid; i <= mid; i++) {
            for (int j = -mid; j <= mid; j++) {
                int pxl = d_filteredImage[(r+i)*cols + (c+j)];
                gx += pxl * x_filter[(i+mid)*3 + (mid+j)];
                gy += pxl * y_filter[(i+mid)*3 + (mid+j)];
            }
        }
        __syncthreads();
        gradient = sqrtf(powf(gx, 2.0) + powf(gy, 2.0));
        angle = atan2f(gy, gx);
        angle = angle/3.14159 * 180;

        if (gradient > 255.0) gradient = 255.0;
        if (gradient <= 0) gradient = 0.0;

        if ((angle >= -22.5 && angle < 22.5) || (angle <= -157.5 || angle > 157.5)) angle = 0;
	else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) angle = 45;
        else if ((angle >= 67.5 && 112.5) || (angle >= -112.5 && angle < -67.5)) angle = 90;
        else angle = 135;

        d_gradientAngle[idx] = (unsigned char) angle;
        d_imageGradient[idx] = (unsigned char) gradient;
    }
}

__global__ void convolve_2d(unsigned char *d_inputImage, unsigned char *d_filteredImage, int rows, int cols, float *d_filter, int filter_rows, int filter_cols, float normalizer) {

    float sum = 0;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    int mid = filter_rows / 2;
  
    if (r >= rows || c >= cols) {
        return;
    }
    
    if ( r < mid || r >= rows-mid || c < mid || c >= cols-mid) {
        d_filteredImage[r*cols+c] = (unsigned char) 0;
    } else {
        for (int i = -mid; i <= mid; i++) {
            for (int j = -mid; j <= mid; j++) {
                int pxl = d_inputImage[(r+i)*cols + (c+j)];
                //sum += pxl * mx[i+mid][j+mid];
                //sum += pxl * nx[(i+mid)*5 + (mid+j)];
                sum += pxl * d_filter[(i+mid)*filter_cols + (mid+j)];
            }
        }
        __syncthreads();
        sum = abs(sum) / normalizer;
        if (sum > 255) sum = 255;
        if (sum < 0) sum = 0;
        d_filteredImage[r*cols+c] = (unsigned char) sum;
    }
}

void sobel_operator(unsigned char *d_filteredImage, unsigned char *d_imageGradient, unsigned char *d_gradientAngle, int rows, int cols) {
    int block_size = 16;

    const dim3 blockSize(block_size, block_size, 1);
    int xCount = rows / block_size + 1;
    int yCount = cols / block_size + 1;
    const dim3 gridSize(xCount, yCount, 1);

    printf("About to launch sobel kernel on GPU\n");
    sobel_kernel<<<gridSize, blockSize>>>(d_filteredImage, d_imageGradient, d_gradientAngle, rows, cols, 30);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}

void non_maxima_suppression(unsigned char *d_imageGradient, unsigned char *d_gradientAngle, unsigned char *d_outputImage, int rows, int cols) {
    int block_size = 16;

    const dim3 blockSize(block_size, block_size, 1);
    int xCount = rows / block_size + 1;
    int yCount = cols / block_size + 1;
    const dim3 gridSize(xCount, yCount, 1);

    printf("About to launch non maxima suppression kernel\n");
    nms_kernel<<<gridSize, blockSize>>>(d_imageGradient, d_gradientAngle, d_outputImage, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}    
    
void gaussian_blur(unsigned char *d_inputImage, unsigned char *d_filteredImage, int rows, int cols) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    int block_size = 16;
    
    const dim3 blockSize(block_size, block_size, 1);
    int xCount = rows / block_size + 1;
    int yCount = cols / block_size + 1;
    const dim3 gridSize(xCount, yCount, 1);

    float h_filter[25] = {2.0, 4.0, 5.0, 4.0, 2.0, 4.0, 9.0, 12.0, 9.0, 4.0, 5.0, 12.0, 15.0, 12.0, 5.0, 4.0, 9.0, 12.0, 9.0, 4.0, 2.0, 4.0, 5.0, 4.0, 2.0};
    float *d_filter;
    cudaMalloc((void**)&d_filter, sizeof(float)*25);
    cudaMemcpy(d_filter, h_filter, sizeof(float)*25, cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    printf("rows= %d; cols= %d\n", rows, cols);

    //img_kernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, rows, cols);
    printf("About to lauch gaussian kernel on GPU\n");
    convolve_2d<<<gridSize, blockSize>>>(d_inputImage, d_filteredImage, rows, cols, d_filter, 5, 5, 159.0);
    
    cudaFree(d_filter);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    printf("Gaussian kernel completed\n");
}

void detect_edges(unsigned char *d_inputImage, unsigned char *d_outputImage, int rows, int cols) {
    unsigned char *d_filteredImage, *d_imageGradient, *d_gradientAngle;
    
    cudaMalloc((void**)&d_filteredImage, sizeof(unsigned char) * rows * cols);
    cudaMalloc((void**)&d_imageGradient, sizeof(unsigned char) * rows * cols);
    cudaMalloc((void**)&d_gradientAngle, sizeof(unsigned char) * rows * cols);

    gaussian_blur(d_inputImage, d_filteredImage, rows, cols);
    sobel_operator(d_filteredImage, d_imageGradient, d_gradientAngle, rows, cols);
    non_maxima_suppression(d_imageGradient, d_gradientAngle, d_outputImage, rows, cols);

    cudaFree(d_filteredImage);
    cudaFree(d_imageGradient);
}


int main(int argc, char **argv) {
    unsigned char *h_inputImage, *d_inputImage, *h_outputImage, *d_outputImage;

    std::string input_file;
    std::string output_file;

    if (argc < 3) {
        printf("Insufficient arguments supplied. Exiting.\n");
        exit(1);
    }

    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);

    cudaFree(0);

    cv::Mat input_image;
    input_image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (input_image.empty()) {
        printf("Failed to open input image\n");
        exit(1);
    }

    cv::Mat output_image;
    output_image.create(input_image.rows, input_image.cols, CV_8UC1);

    h_inputImage = input_image.ptr<unsigned char>(0);
    h_outputImage = output_image.ptr<unsigned char>(0);

    const size_t pixelCount = input_image.rows * input_image.cols;

    cudaMalloc((void**)&d_inputImage, sizeof(unsigned char) * pixelCount);
    cudaMalloc((void**)&d_outputImage, sizeof(unsigned char) * pixelCount);
    //cudaMemset(*d_outputImage, 0, pixelCount * sizeof(unsigned char));
    //h_outputImage = (unsigned char*) malloc(sizeof(unsigned char) * pixelCount);

    cudaMemcpy(d_inputImage, h_inputImage, sizeof(unsigned char) * pixelCount, cudaMemcpyHostToDevice);

    //launch_kernel(d_inputImage, d_outputImage, input_image.rows, input_image.cols);
    detect_edges(d_inputImage, d_outputImage, input_image.rows, input_image.cols);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    

    cudaMemcpy(h_outputImage, d_outputImage, sizeof(unsigned char) * pixelCount, cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    cv::Mat output(input_image.rows, input_image.cols, CV_8UC1, (void*)h_outputImage);
    cv::imwrite(output_file.c_str(), output);

    printf("Done\n");

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
