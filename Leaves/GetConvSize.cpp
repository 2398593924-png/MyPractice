#include <stdio.h>
#include <math.h>

int ConvSize(int input_size, int kernel_size, int padding, int stride){
    double tmp = (input_size - kernel_size + 2 * padding) / stride;
    return floor(tmp) + 1;
}

int TConvSize(int input_size, int kernel_size, int padding, int stride){
    double tmp = (input_size - 1) * stride + kernel_size - 2.0 * padding;
    return ceil(tmp);
}

int main(){
    int n, input_size, kernel_size, padding, stride;
    while(true){
        printf("ÆÕÍ¨0, ×ªÖÃ1: ");
        scanf("%d", &n);
        printf("ÒÀ´ÎÊäÈë:input_size, K, P, S:");
        scanf("%d %d %d %d", &input_size, &kernel_size, &padding, &stride);
        if (n == 0){
            printf("³ß´ç:%d", ConvSize(input_size, kernel_size, padding, stride));
        }else{
            printf("³ß´ç:%d", TConvSize(input_size, kernel_size, padding, stride));
        }
        system("pause>nul");
        system("cls");
    }
    return 0;
}