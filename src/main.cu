#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

double copy_total = 0.0;
double malloc_total = 0.0;
double free_total = 0.0;
int BLOCK_SIZE = 16;

clock_t main_start, main_end, read_start, read_end, write_start, write_end, in_copy_start, in_copy_end, 
        out_copy_start, out_copy_end, detect_start, detect_end, gauss_start, gauss_end,
        sobel_start, sobel_end, nms_start, nms_end, hysteresis_start, hysteresis_end;

__global__ void img_kernel(unsigned char *d_inputImage, unsigned char *d_outputImage, int rows, int cols) {
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (y < cols && x < rows) {
        int idx = x * cols + y;
        d_outputImage[idx] = d_inputImage[idx];
    }
    return;
}

__global__ void edge_grow_kernel(unsigned char *d_edgeMask, unsigned char *d_outputImage, bool *d_done, int rows, int cols) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = r*cols+c, neighborOne, neighborTwo, neighborThree, neighborFour, neighborFive, neighborSix, neighborSeven, neighborEight;
    if (r >= rows || c >= cols) {
        return;
    }

    if (r < 1 || r >= rows-1 || c < 1 || c >= cols-1) {
        d_outputImage[idx] = (unsigned char) 0;
    } else if (d_edgeMask[idx] == 2) {
                neighborOne = idx - 1; neighborTwo = idx + 1;
                neighborThree = (r-1)*cols+c-1; neighborFour = (r+1)*cols+c+1;
                neighborFive = idx - cols; neighborSix = idx + cols;
                neighborSeven = (r-1)*cols+c+1; neighborEight = (r+1)*cols+c-1;

        if (d_edgeMask[neighborOne] == 1) {
            d_edgeMask[neighborOne] = (unsigned char) 2; 
            d_outputImage[neighborOne] = (unsigned char) 255;
            *d_done = false;
        }
        if (d_edgeMask[neighborTwo] == 1) {
            d_edgeMask[neighborTwo] = (unsigned char) 2;
            d_outputImage[neighborTwo] = (unsigned char) 255;
            *d_done = false;
        }
        if (d_edgeMask[neighborThree] == 1) {
            d_edgeMask[neighborThree] = (unsigned char) 2; 
            d_outputImage[neighborThree] = (unsigned char) 255;
            *d_done = false;
        }
        if (d_edgeMask[neighborFour] == 1) {
            d_edgeMask[neighborFour] = (unsigned char) 2;
            d_outputImage[neighborFour] = (unsigned char) 255;
            *d_done = false;
        }
        if (d_edgeMask[neighborFive] == 1) {
            d_edgeMask[neighborFive] = (unsigned char) 2; 
            d_outputImage[neighborFive] = (unsigned char) 255;
            *d_done = false;
        }
        if (d_edgeMask[neighborSix] == 1) {
            d_edgeMask[neighborSix] = (unsigned char) 2;
            d_outputImage[neighborSix] = (unsigned char) 255;
            *d_done = false;
        }
        if (d_edgeMask[neighborSeven] == 1) {
            d_edgeMask[neighborSeven] = (unsigned char) 2; 
            d_outputImage[neighborSeven] = (unsigned char) 255;
            *d_done = false;
        }
        if (d_edgeMask[neighborEight] == 1) {
            d_edgeMask[neighborEight] = (unsigned char) 2;
            d_outputImage[neighborEight] = (unsigned char) 255;
            *d_done = false;
        }
    }
}


__global__ void nms_kernel(unsigned char *d_imageGradient, unsigned char *d_gradientAngle, unsigned char *d_edgeMask, unsigned char *d_outputImage, int rows, int cols, int high_threshold, int low_threshold) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = r*cols+c, neighborOne = idx, neighborTwo = idx;
    if (r >= rows || c >= cols) {
        return;
    }

    if (r < 1 || r >= rows-1 || c < 1 || c >= cols-1) {
        d_edgeMask[idx] = (unsigned char) 0;
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
            if (d_imageGradient[idx] > high_threshold) {
            d_edgeMask[idx] = (unsigned char) 2;
            d_outputImage[idx] = (unsigned char) 255;
            } else if (d_imageGradient[idx] > low_threshold) {
                d_edgeMask[idx] = (unsigned char) 1;
                d_outputImage[idx] = (unsigned char) 0;
            } else {
                d_edgeMask[idx] = (unsigned char) 0;
                d_outputImage[idx] = (unsigned char) 0;
            }
        } else {
            d_edgeMask[idx] = (unsigned char) 0;
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
        //gx = gx/8; gy = gy/8;
        gradient = sqrtf(powf(gx, 2.0) + powf(gy, 2.0));
        angle = atan2f(gy, gx);
        angle = angle/3.14159 * 180;

/*        if (gradient > 255.0) gradient = 255.0;
        if (gradient <= 0) gradient = 0.0;
*/
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

__global__ void gaussian_kernel(unsigned char *d_inputImage, unsigned char *d_filteredImage, int rows, int cols) {
    float filter[25] = {2.0, 4.0, 5.0, 4.0, 2.0, 4.0, 9.0, 12.0, 9.0, 4.0, 5.0, 12.0, 15.0, 12.0, 5.0, 4.0, 9.0, 12.0, 9.0, 4.0, 2.0, 4.0, 5.0, 4.0, 2.0};
    float normalizer = 159.0;
    float sum = 0;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    int mid = 2;
  
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
                sum += pxl * filter[(i+mid)*5 + (mid+j)];
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
    int block_size = BLOCK_SIZE;

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

void grow_edges(unsigned char *d_edgeMask, unsigned char *d_outputImage, bool *d_done, int rows, int cols) {
    int block_size = BLOCK_SIZE;

    const dim3 blockSize(block_size, block_size, 1);
    int xCount = rows / block_size + 1;
    int yCount = cols / block_size + 1;
    const dim3 gridSize(xCount, yCount, 1);

    //printf("About to launch edge grow kernel\n");
    edge_grow_kernel<<<gridSize, blockSize>>>(d_edgeMask, d_outputImage, d_done, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
} 

void non_maxima_suppression(unsigned char *d_imageGradient, unsigned char *d_gradientAngle, unsigned char *d_edgeMask, unsigned char *d_outputImage, int rows, int cols, int high_threshold, int low_threshold) {
    int block_size = BLOCK_SIZE;

    const dim3 blockSize(block_size, block_size, 1);
    int xCount = rows / block_size + 1;
    int yCount = cols / block_size + 1;
    const dim3 gridSize(xCount, yCount, 1);

    printf("About to launch non maxima suppression kernel\n");
    nms_kernel<<<gridSize, blockSize>>>(d_imageGradient, d_gradientAngle, d_edgeMask, d_outputImage, rows, cols, high_threshold, low_threshold);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}    
    
void gaussian_blur(unsigned char *d_inputImage, unsigned char *d_filteredImage, int rows, int cols) {
    int block_size = BLOCK_SIZE;
    
    const dim3 blockSize(block_size, block_size, 1);
    int xCount = rows / block_size + 1;
    int yCount = cols / block_size + 1;
    const dim3 gridSize(xCount, yCount, 1);

    printf("About to lauch gaussian kernel on GPU\n");
    gaussian_kernel<<<gridSize, blockSize>>>(d_inputImage, d_filteredImage, rows, cols);
    
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}

void detect_edges(unsigned char *d_inputImage, unsigned char *d_outputImage, int rows, int cols, int high_threshold, int low_threshold) {
    unsigned char *d_filteredImage, *d_imageGradient, *d_gradientAngle, *d_edgeMask;
    bool *d_done;
    bool h_done = true;
    clock_t malloc_start = clock();
    cudaMalloc((void**)&d_filteredImage, sizeof(unsigned char) * rows * cols);
    cudaMalloc((void**)&d_imageGradient, sizeof(unsigned char) * rows * cols);
    cudaMalloc((void**)&d_gradientAngle, sizeof(unsigned char) * rows * cols);
    cudaMalloc((void**)&d_edgeMask, sizeof(unsigned char) * rows * cols);
    cudaMalloc((void**)&d_done, sizeof(bool));
    clock_t malloc_end = clock();
    malloc_total += (double) (malloc_end - malloc_start) / CLOCKS_PER_SEC;

    gauss_start = clock();
    gaussian_blur(d_inputImage, d_filteredImage, rows, cols);
    gauss_end = clock();

    sobel_start = clock();
    sobel_operator(d_filteredImage, d_imageGradient, d_gradientAngle, rows, cols);
    sobel_end = clock();

    nms_start = clock();
    non_maxima_suppression(d_imageGradient, d_gradientAngle, d_edgeMask, d_outputImage, rows, cols, high_threshold, low_threshold);
    nms_end = clock();
   
    hysteresis_start = clock();
    int count = 0;
    do {
        count++;
        h_done = true;
        cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
        //kernel
        grow_edges(d_edgeMask, d_outputImage, d_done, rows, cols);
        cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (!h_done);
    printf("Completed hysteresis in %d rounds\n", count);
/*
    for (int i = 0; i < 4; i++)
        grow_edges(d_edgeMask, d_outputImage, rows, cols);
*/
    hysteresis_end = clock();

    clock_t free_start = clock();
    cudaFree(d_filteredImage);
    cudaFree(d_imageGradient);
    cudaFree(d_gradientAngle);
    cudaFree(d_edgeMask);
    cudaFree(d_done);
    clock_t free_end = clock();
    free_total += (double) (free_end - free_start) / CLOCKS_PER_SEC;
}


int main(int argc, char **argv) {
    unsigned char *h_inputImage, *d_inputImage, *h_outputImage, *d_outputImage;

    main_start = clock();

    std::string input_file;
    std::string output_file;

    int high_threshold, low_threshold;

    if (argc < 5) {
        printf("Insufficient arguments supplied. Exiting.\n");
        exit(1);
    }

    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);

    high_threshold = atoi(argv[3]);
    low_threshold = atoi(argv[4]);

    cudaFree(0);

    read_start = clock();
    cv::Mat input_image;
    input_image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (input_image.empty()) {
        printf("Failed to open input image\n");
        exit(1);
    }
    read_end = clock();

    cv::Mat output_image;
    output_image.create(input_image.rows, input_image.cols, CV_8UC1);

    h_inputImage = input_image.ptr<unsigned char>(0);
    h_outputImage = output_image.ptr<unsigned char>(0);

    const size_t pixelCount = input_image.rows * input_image.cols;

    clock_t malloc_start = clock();
    cudaMalloc((void**)&d_inputImage, sizeof(unsigned char) * pixelCount);
    cudaMalloc((void**)&d_outputImage, sizeof(unsigned char) * pixelCount);
    clock_t malloc_end = clock();
    malloc_total += (double) (malloc_end - malloc_start) / CLOCKS_PER_SEC;

    in_copy_start = clock();
    cudaMemcpy(d_inputImage, h_inputImage, sizeof(unsigned char) * pixelCount, cudaMemcpyHostToDevice);
    in_copy_end = clock();

    //launch_kernel(d_inputImage, d_outputImage, input_image.rows, input_image.cols);
    detect_start = clock();
    detect_edges(d_inputImage, d_outputImage, input_image.rows, input_image.cols, high_threshold, low_threshold);
    detect_end = clock();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    
    out_copy_start = clock();
    cudaMemcpy(h_outputImage, d_outputImage, sizeof(unsigned char) * pixelCount, cudaMemcpyDeviceToHost);
    out_copy_end = clock();

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    write_start = clock();
    cv::Mat output(input_image.rows, input_image.cols, CV_8UC1, (void*)h_outputImage);
    cv::imwrite(output_file.c_str(), output);
    write_end = clock();

    printf("Done\n");

    clock_t free_start = clock();
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    clock_t free_end = clock();
    free_total += (double) (free_end - free_start) / CLOCKS_PER_SEC;

    copy_total += (double) (in_copy_end + out_copy_end - in_copy_start - out_copy_start) / CLOCKS_PER_SEC;

    main_end = clock();
    printf("\nProfiling Summary:\n==================================================\n");
    printf("Time spent in reading input image: %lf\n", (double) (read_end - read_start) / CLOCKS_PER_SEC);
    printf("Time spent in detect_edges: %lf\n", (double) (detect_end - detect_start) / CLOCKS_PER_SEC);
    printf("|-- Time spent in gaussian_blur: %lf\n", (double) (gauss_end - gauss_start) / CLOCKS_PER_SEC);
    printf("|-- Time spent in sobel_operator: %lf\n", (double) (sobel_end - sobel_start) / CLOCKS_PER_SEC);
    printf("|-- Time spent in non_maxima_suppression: %lf\n", (double) (nms_end - nms_start) / CLOCKS_PER_SEC);
    printf("|-- Time spent in hysteresis: %lf\n", (double) (hysteresis_end - hysteresis_start) / CLOCKS_PER_SEC);
    printf("Time spent in writing output image: %lf\n", (double) (write_end - write_start) / CLOCKS_PER_SEC);
    printf("Time spent in main : %lf\n", (double) (main_end - main_start) / CLOCKS_PER_SEC);
    printf("--------------------------------------------------\n");
    printf("Time spent in cudaMalloc calls: %lf\n", malloc_total);
    printf("Time spent in cudaMemcpy calls: %lf\n", copy_total);
    printf("Time spent in cudaFree calls: %lf\n", free_total);

    return 0;
}
