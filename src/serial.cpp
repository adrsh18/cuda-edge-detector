#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>

double malloc_total = 0.0, free_total = 0.0;
clock_t main_start, main_end, read_start, read_end, write_start, write_end, detect_start, detect_end, gauss_start, gauss_end,
        sobel_start, sobel_end, nms_start, nms_end, hysteresis_start, hysteresis_end;

void edge_grow_kernel(unsigned char *h_edgeMask, unsigned char *h_outputImage, int rows, int cols, int r, int c) {

    int idx = r*cols+c, neighborOne, neighborTwo, neighborThree, neighborFour, neighborFive, neighborSix, neighborSeven, neighborEight;
    if (r >= rows || c >= cols) {
        return;
    }

    if (r < 1 || r >= rows-1 || c < 1 || c >= cols-1) {
        h_outputImage[idx] = (unsigned char) 0;
    } else if (h_edgeMask[idx] == 2) {
                neighborOne = idx - 1; neighborTwo = idx + 1;
                neighborThree = (r-1)*cols+c-1; neighborFour = (r+1)*cols+c+1;
                neighborFive = idx - cols; neighborSix = idx + cols;
                neighborSeven = (r-1)*cols+c+1; neighborEight = (r+1)*cols+c-1;

        if (h_edgeMask[neighborOne] == 1) {
            h_edgeMask[neighborOne] = (unsigned char) 2; 
            h_outputImage[neighborOne] = (unsigned char) 255;
        }
        if (h_edgeMask[neighborTwo] == 1) {
            h_edgeMask[neighborTwo] = (unsigned char) 2;
            h_outputImage[neighborTwo] = (unsigned char) 255;
        }
        if (h_edgeMask[neighborThree] == 1) {
            h_edgeMask[neighborThree] = (unsigned char) 2; 
            h_outputImage[neighborThree] = (unsigned char) 255;
        }
        if (h_edgeMask[neighborFour] == 1) {
            h_edgeMask[neighborFour] = (unsigned char) 2;
            h_outputImage[neighborFour] = (unsigned char) 255;
        }
        if (h_edgeMask[neighborFive] == 1) {
            h_edgeMask[neighborFive] = (unsigned char) 2; 
            h_outputImage[neighborFive] = (unsigned char) 255;
        }
        if (h_edgeMask[neighborSix] == 1) {
            h_edgeMask[neighborSix] = (unsigned char) 2;
            h_outputImage[neighborSix] = (unsigned char) 255;
        }
        if (h_edgeMask[neighborSeven] == 1) {
            h_edgeMask[neighborSeven] = (unsigned char) 2; 
            h_outputImage[neighborSeven] = (unsigned char) 255;
        }
        if (h_edgeMask[neighborEight] == 1) {
            h_edgeMask[neighborEight] = (unsigned char) 2;
            h_outputImage[neighborEight] = (unsigned char) 255;
        }
    }
}

void nms_kernel(unsigned char *h_imageGradient, unsigned char *h_gradientAngle, unsigned char *h_edgeMask, unsigned char *h_outputImage, int rows, int cols, int high_threshold, int low_threshold, int r, int c) {

    int idx = r*cols+c, neighborOne = idx, neighborTwo = idx;
    if (r >= rows || c >= cols) {
        return;
    }

    if (r < 1 || r >= rows-1 || c < 1 || c >= cols-1) {
        h_edgeMask[idx] = (unsigned char) 0;
        h_outputImage[idx] = (unsigned char) 0;
    } else {
        switch(h_gradientAngle[idx]) {
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
        if (h_imageGradient[idx] > h_imageGradient[neighborOne] && h_imageGradient[idx] > h_imageGradient[neighborTwo]) {
            if (h_imageGradient[idx] > high_threshold) {
            h_edgeMask[idx] = (unsigned char) 2;
            h_outputImage[idx] = (unsigned char) 255;
            } else if (h_imageGradient[idx] > low_threshold) {
                h_edgeMask[idx] = (unsigned char) 1;
                h_outputImage[idx] = (unsigned char) 0;
            } else {
                h_edgeMask[idx] = (unsigned char) 0;
                h_outputImage[idx] = (unsigned char) 0;
            }
        } else {
            h_edgeMask[idx] = (unsigned char) 0;
            h_outputImage[idx] = (unsigned char) 0;
        }
    }
}

void sobel_kernel(unsigned char *h_filteredImage, unsigned char *h_imageGradient, unsigned char *h_gradientAngle, int rows, int cols, int threshold, int r, int c) {
    int x_filter[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int y_filter[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    float gradient = 0.0, gx = 0.0, gy = 0.0, angle = 0.0;

    int mid = 1, idx = r*cols+c;

    if (r >= rows || c >= cols) {
         return;
    }

    if (r < mid || r >= rows-mid || c < mid || c >= cols-mid) {
        h_imageGradient[idx] = (unsigned char) 0;
        h_gradientAngle[idx] = (unsigned char) 0;
    } else {
        for (int i = -mid; i <= mid; i++) {
            for (int j = -mid; j <= mid; j++) {
                int pxl = h_filteredImage[(r+i)*cols + (c+j)];
                gx += pxl * x_filter[(i+mid)*3 + (mid+j)];
                gy += pxl * y_filter[(i+mid)*3 + (mid+j)];
            }
        }
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

        h_gradientAngle[idx] = (unsigned char) angle;
        h_imageGradient[idx] = (unsigned char) gradient;
    }
}

void convolve_2d(unsigned char *h_inputImage, unsigned char *h_filteredImage, int rows, int cols, float *h_filter, int filter_rows, int filter_cols, float normalizer, int r, int c) {

    float sum = 0;

    int mid = filter_rows / 2;
  
    if (r >= rows || c >= cols) {
        return;
    }
    
    if ( r < mid || r >= rows-mid || c < mid || c >= cols-mid) {
        h_filteredImage[r*cols+c] = (unsigned char) 0;
    } else {
        for (int i = -mid; i <= mid; i++) {
            for (int j = -mid; j <= mid; j++) {
                int pxl = h_inputImage[(r+i)*cols + (c+j)];
                //sum += pxl * mx[i+mid][j+mid];
                //sum += pxl * nx[(i+mid)*5 + (mid+j)];
                sum += pxl * h_filter[(i+mid)*filter_cols + (mid+j)];
            }
        }
        sum = abs(sum) / normalizer;
        if (sum > 255) sum = 255;
        if (sum < 0) sum = 0;
        h_filteredImage[r*cols+c] = (unsigned char) sum;
    }
}

void sobel_operator(unsigned char *h_filteredImage, unsigned char *h_imageGradient, unsigned char *h_gradientAngle, int rows, int cols) {

    printf("About to run sobel operator on CPU\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sobel_kernel(h_filteredImage, h_imageGradient, h_gradientAngle, rows, cols, 30, i, j);
        }
    }
}

void grow_edges(unsigned char *h_edgeMask, unsigned char *h_outputImage, int rows, int cols) {

    printf("About to grow edges on CPU\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            edge_grow_kernel(h_edgeMask, h_outputImage, rows, cols, i, j);
        }
    }
} 

void non_maxima_suppression(unsigned char *h_imageGradient, unsigned char *h_gradientAngle, unsigned char *h_edgeMask, unsigned char *h_outputImage, int rows, int cols, int high_threshold, int low_threshold) {

    printf("About to run non maxima suppression on CPU\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nms_kernel(h_imageGradient, h_gradientAngle, h_edgeMask, h_outputImage, rows, cols, high_threshold, low_threshold, i, j);
        }
    }

}    
    
void gaussian_blur(unsigned char *h_inputImage, unsigned char *h_filteredImage, int rows, int cols) {
    
    float h_filter[25] = {2.0, 4.0, 5.0, 4.0, 2.0, 4.0, 9.0, 12.0, 9.0, 4.0, 5.0, 12.0, 15.0, 12.0, 5.0, 4.0, 9.0, 12.0, 9.0, 4.0, 2.0, 4.0, 5.0, 4.0, 2.0};
    
    //img_kernel<<<gridSize, blockSize>>>(h_inputImage, h_outputImage, rows, cols);
    printf("About to run gaussian blur on CPU\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            convolve_2d(h_inputImage, h_filteredImage, rows, cols, h_filter, 5, 5, 159.0, i, j);
        }
    }
}

void detect_edges(unsigned char *h_inputImage, unsigned char *h_outputImage, int rows, int cols, int high_threshold, int low_threshold) {
    unsigned char *h_filteredImage, *h_imageGradient, *h_gradientAngle, *h_edgeMask;
    
    clock_t malloc_start = clock();
    h_filteredImage = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols);
    h_imageGradient = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols);
    h_gradientAngle = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols);
    h_edgeMask = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols);
    clock_t malloc_end = clock();
    malloc_total += (double) (malloc_end - malloc_start) / CLOCKS_PER_SEC;

    gauss_start = clock();
    gaussian_blur(h_inputImage, h_filteredImage, rows, cols);
    gauss_end = clock();

    sobel_start = clock();
    sobel_operator(h_filteredImage, h_imageGradient, h_gradientAngle, rows, cols);
    sobel_end = clock();

    nms_start = clock();
    non_maxima_suppression(h_imageGradient, h_gradientAngle, h_edgeMask, h_outputImage, rows, cols, high_threshold, low_threshold);
    nms_end = clock();

    hysteresis_start = clock();
    for (int i = 0; i < 2; i++)
        grow_edges(h_edgeMask, h_outputImage, rows, cols);
    hysteresis_end = clock();

    clock_t free_start = clock();
    free(h_filteredImage);
    free(h_imageGradient);
    free(h_gradientAngle);
    free(h_edgeMask);
    clock_t free_end = clock();
    free_total += (double) (free_end - free_start) / CLOCKS_PER_SEC;
}


int main(int argc, char **argv) {
    unsigned char *h_inputImage, *h_outputImage;

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

    read_start = clock();
    cv::Mat input_image;
    input_image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (input_image.empty()) {
        printf("Failed to open input image\n");
        exit(1);
    }
    read_end = clock();

    clock_t malloc_start = clock();
    h_inputImage = input_image.ptr<unsigned char>(0);
    h_outputImage = (unsigned char*) malloc(sizeof(unsigned char) * input_image.rows * input_image.cols);
    clock_t malloc_end = clock();
    malloc_total += (double) (malloc_end - malloc_start) / CLOCKS_PER_SEC;

    detect_start = clock();
    detect_edges(h_inputImage, h_outputImage, input_image.rows, input_image.cols, high_threshold, low_threshold);
    detect_end = clock();

    write_start = clock();
    cv::Mat output(input_image.rows, input_image.cols, CV_8UC1, (void*)h_outputImage);
    cv::imwrite(output_file.c_str(), output);
    write_end = clock();

    printf("Done\n");
    free(h_outputImage);

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
    printf("Time spent in malloc calls: %lf\n", malloc_total);
    printf("Time spent in free calls: %lf\n", free_total);

    return 0;
}
