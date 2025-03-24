#include "yolo5n.h"
#include "yolo5n_weights_0.h"
#include "yolo5n_weights_1.h"
#include "yolo5n_weights_2.h"
#include "yolo5n_weights_3.h"
#include "yolo5n_weights_4.h"
#include "yolo5n_weights_5.h"
#include "yolo5n_weights_6.h"
#include "yolo5n_weights_7.h"
#include "yolo5n_weights_8.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "esp32-hal-psram.h"


/*  Operator: Conv 
    Name in model: /model.0/conv/Conv
    Input: images [1, 3, 224, 224]
    Input: model_0_conv_weight [16, 3, 6, 6]
    Input: model_0_conv_bias [16]
    Output: _model_0_conv_Conv_output_0 [1, 16, 112, 112]
*/
void node__model_0_conv_Conv(const float images[1][3][224][224], const float model_0_conv_weight[16][3][6][6], const float model_0_conv_bias[16], float _model_0_conv_Conv_output_0[1][16][112][112]) {
    // Dimension constants
    const int N = 1, C_in = 3, H_in = 224, W_in = 224;
    const int K = 16, K_h = 6, K_w = 6;
    const int H_out = 112, W_out = 112;
    // Convolution parameters
    const int stride_h = 2, stride_w = 2;
    const int pad_t = 2, pad_b = 2, pad_l = 2, pad_r = 2;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_0_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += images[n][c][h_in][w_in] * model_0_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_0_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.0/act/Sigmoid
    Input: _model_0_conv_Conv_output_0 [1, 16, 112, 112]
    Output: _model_0_act_Sigmoid_output_0 [1, 16, 112, 112]
*/
void node__model_0_act_Sigmoid(const float _model_0_conv_Conv_output_0[1][16][112][112], float _model_0_act_Sigmoid_output_0[1][16][112][112]) {
    float *X_ptr = (float *)_model_0_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_0_act_Sigmoid_output_0;
    for (int i = 0; i < 200704; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.0/act/Mul
    Input: _model_0_conv_Conv_output_0 [1, 16, 112, 112]
    Input: _model_0_act_Sigmoid_output_0 [1, 16, 112, 112]
    Output: _model_0_act_Mul_output_0 [1, 16, 112, 112]
*/
void node__model_0_act_Mul(const float _model_0_conv_Conv_output_0[1][16][112][112], const float _model_0_act_Sigmoid_output_0[1][16][112][112], float _model_0_act_Mul_output_0[1][16][112][112]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 16; d1++) {
    for (int d2 = 0; d2 < 112; d2++) {
    for (int d3 = 0; d3 < 112; d3++) {
        _model_0_act_Mul_output_0[d0][d1][d2][d3] = _model_0_conv_Conv_output_0[d0][d1][d2][d3] * _model_0_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.1/conv/Conv
    Input: _model_0_act_Mul_output_0 [1, 16, 112, 112]
    Input: model_1_conv_weight [32, 16, 3, 3]
    Input: model_1_conv_bias [32]
    Output: _model_1_conv_Conv_output_0 [1, 32, 56, 56]
*/
void node__model_1_conv_Conv(const float _model_0_act_Mul_output_0[1][16][112][112], const float model_1_conv_weight[32][16][3][3], const float model_1_conv_bias[32], float _model_1_conv_Conv_output_0[1][32][56][56]) {
    // Dimension constants
    const int N = 1, C_in = 16, H_in = 112, W_in = 112;
    const int K = 32, K_h = 3, K_w = 3;
    const int H_out = 56, W_out = 56;
    // Convolution parameters
    const int stride_h = 2, stride_w = 2;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_1_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_0_act_Mul_output_0[n][c][h_in][w_in] * model_1_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_1_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.1/act/Sigmoid
    Input: _model_1_conv_Conv_output_0 [1, 32, 56, 56]
    Output: _model_1_act_Sigmoid_output_0 [1, 32, 56, 56]
*/
void node__model_1_act_Sigmoid(const float _model_1_conv_Conv_output_0[1][32][56][56], float _model_1_act_Sigmoid_output_0[1][32][56][56]) {
    float *X_ptr = (float *)_model_1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_1_act_Sigmoid_output_0;
    for (int i = 0; i < 100352; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.1/act/Mul
    Input: _model_1_conv_Conv_output_0 [1, 32, 56, 56]
    Input: _model_1_act_Sigmoid_output_0 [1, 32, 56, 56]
    Output: _model_1_act_Mul_output_0 [1, 32, 56, 56]
*/
void node__model_1_act_Mul(const float _model_1_conv_Conv_output_0[1][32][56][56], const float _model_1_act_Sigmoid_output_0[1][32][56][56], float _model_1_act_Mul_output_0[1][32][56][56]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 56; d2++) {
    for (int d3 = 0; d3 < 56; d3++) {
        _model_1_act_Mul_output_0[d0][d1][d2][d3] = _model_1_conv_Conv_output_0[d0][d1][d2][d3] * _model_1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.2/cv1/conv/Conv
    Input: _model_1_act_Mul_output_0 [1, 32, 56, 56]
    Input: model_2_cv1_conv_weight [16, 32, 1, 1]
    Input: model_2_cv1_conv_bias [16]
    Output: _model_2_cv1_conv_Conv_output_0 [1, 16, 56, 56]
*/
void node__model_2_cv1_conv_Conv(const float _model_1_act_Mul_output_0[1][32][56][56], const float model_2_cv1_conv_weight[16][32][1][1], const float model_2_cv1_conv_bias[16], float _model_2_cv1_conv_Conv_output_0[1][16][56][56]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 56, W_in = 56;
    const int K = 16, K_h = 1, K_w = 1;
    const int H_out = 56, W_out = 56;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_2_cv1_conv_bias[k];
                float ker_val = model_2_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_2_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_2_cv1_conv_Conv_output_0[n][k][h][w] += _model_1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.2/cv2/conv/Conv
    Input: _model_1_act_Mul_output_0 [1, 32, 56, 56]
    Input: model_2_cv2_conv_weight [16, 32, 1, 1]
    Input: model_2_cv2_conv_bias [16]
    Output: _model_2_cv2_conv_Conv_output_0 [1, 16, 56, 56]
*/
void node__model_2_cv2_conv_Conv(const float _model_1_act_Mul_output_0[1][32][56][56], const float model_2_cv2_conv_weight[16][32][1][1], const float model_2_cv2_conv_bias[16], float _model_2_cv2_conv_Conv_output_0[1][16][56][56]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 56, W_in = 56;
    const int K = 16, K_h = 1, K_w = 1;
    const int H_out = 56, W_out = 56;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_2_cv2_conv_bias[k];
                float ker_val = model_2_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_2_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_2_cv2_conv_Conv_output_0[n][k][h][w] += _model_1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.2/cv1/act/Sigmoid
    Input: _model_2_cv1_conv_Conv_output_0 [1, 16, 56, 56]
    Output: _model_2_cv1_act_Sigmoid_output_0 [1, 16, 56, 56]
*/
void node__model_2_cv1_act_Sigmoid(const float _model_2_cv1_conv_Conv_output_0[1][16][56][56], float _model_2_cv1_act_Sigmoid_output_0[1][16][56][56]) {
    float *X_ptr = (float *)_model_2_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_2_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 50176; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.2/cv2/act/Sigmoid
    Input: _model_2_cv2_conv_Conv_output_0 [1, 16, 56, 56]
    Output: _model_2_cv2_act_Sigmoid_output_0 [1, 16, 56, 56]
*/
void node__model_2_cv2_act_Sigmoid(const float _model_2_cv2_conv_Conv_output_0[1][16][56][56], float _model_2_cv2_act_Sigmoid_output_0[1][16][56][56]) {
    float *X_ptr = (float *)_model_2_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_2_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 50176; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.2/cv1/act/Mul
    Input: _model_2_cv1_conv_Conv_output_0 [1, 16, 56, 56]
    Input: _model_2_cv1_act_Sigmoid_output_0 [1, 16, 56, 56]
    Output: _model_2_cv1_act_Mul_output_0 [1, 16, 56, 56]
*/
void node__model_2_cv1_act_Mul(const float _model_2_cv1_conv_Conv_output_0[1][16][56][56], const float _model_2_cv1_act_Sigmoid_output_0[1][16][56][56], float _model_2_cv1_act_Mul_output_0[1][16][56][56]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 16; d1++) {
    for (int d2 = 0; d2 < 56; d2++) {
    for (int d3 = 0; d3 < 56; d3++) {
        _model_2_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_2_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_2_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.2/cv2/act/Mul
    Input: _model_2_cv2_conv_Conv_output_0 [1, 16, 56, 56]
    Input: _model_2_cv2_act_Sigmoid_output_0 [1, 16, 56, 56]
    Output: _model_2_cv2_act_Mul_output_0 [1, 16, 56, 56]
*/
void node__model_2_cv2_act_Mul(const float _model_2_cv2_conv_Conv_output_0[1][16][56][56], const float _model_2_cv2_act_Sigmoid_output_0[1][16][56][56], float _model_2_cv2_act_Mul_output_0[1][16][56][56]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 16; d1++) {
    for (int d2 = 0; d2 < 56; d2++) {
    for (int d3 = 0; d3 < 56; d3++) {
        _model_2_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_2_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_2_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.2/m/m.0/cv1/conv/Conv
    Input: _model_2_cv1_act_Mul_output_0 [1, 16, 56, 56]
    Input: model_2_m_0_cv1_conv_weight [16, 16, 1, 1]
    Input: model_2_m_0_cv1_conv_bias [16]
    Output: _model_2_m_m_0_cv1_conv_Conv_output_0 [1, 16, 56, 56]
*/
void node__model_2_m_m_0_cv1_conv_Conv(const float _model_2_cv1_act_Mul_output_0[1][16][56][56], const float model_2_m_0_cv1_conv_weight[16][16][1][1], const float model_2_m_0_cv1_conv_bias[16], float _model_2_m_m_0_cv1_conv_Conv_output_0[1][16][56][56]) {
    // Dimension constants
    const int N = 1, C_in = 16, H_in = 56, W_in = 56;
    const int K = 16, K_h = 1, K_w = 1;
    const int H_out = 56, W_out = 56;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_2_m_0_cv1_conv_bias[k];
                float ker_val = model_2_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_2_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_2_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_2_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.2/m/m.0/cv1/act/Sigmoid
    Input: _model_2_m_m_0_cv1_conv_Conv_output_0 [1, 16, 56, 56]
    Output: _model_2_m_m_0_cv1_act_Sigmoid_output_0 [1, 16, 56, 56]
*/
void node__model_2_m_m_0_cv1_act_Sigmoid(const float _model_2_m_m_0_cv1_conv_Conv_output_0[1][16][56][56], float _model_2_m_m_0_cv1_act_Sigmoid_output_0[1][16][56][56]) {
    float *X_ptr = (float *)_model_2_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_2_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 50176; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.2/m/m.0/cv1/act/Mul
    Input: _model_2_m_m_0_cv1_conv_Conv_output_0 [1, 16, 56, 56]
    Input: _model_2_m_m_0_cv1_act_Sigmoid_output_0 [1, 16, 56, 56]
    Output: _model_2_m_m_0_cv1_act_Mul_output_0 [1, 16, 56, 56]
*/
void node__model_2_m_m_0_cv1_act_Mul(const float _model_2_m_m_0_cv1_conv_Conv_output_0[1][16][56][56], const float _model_2_m_m_0_cv1_act_Sigmoid_output_0[1][16][56][56], float _model_2_m_m_0_cv1_act_Mul_output_0[1][16][56][56]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 16; d1++) {
    for (int d2 = 0; d2 < 56; d2++) {
    for (int d3 = 0; d3 < 56; d3++) {
        _model_2_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_2_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_2_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.2/m/m.0/cv2/conv/Conv
    Input: _model_2_m_m_0_cv1_act_Mul_output_0 [1, 16, 56, 56]
    Input: model_2_m_0_cv2_conv_weight [16, 16, 3, 3]
    Input: model_2_m_0_cv2_conv_bias [16]
    Output: _model_2_m_m_0_cv2_conv_Conv_output_0 [1, 16, 56, 56]
*/
void node__model_2_m_m_0_cv2_conv_Conv(const float _model_2_m_m_0_cv1_act_Mul_output_0[1][16][56][56], const float model_2_m_0_cv2_conv_weight[16][16][3][3], const float model_2_m_0_cv2_conv_bias[16], float _model_2_m_m_0_cv2_conv_Conv_output_0[1][16][56][56]) {
    // Dimension constants
    const int N = 1, C_in = 16, H_in = 56, W_in = 56;
    const int K = 16, K_h = 3, K_w = 3;
    const int H_out = 56, W_out = 56;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_2_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_2_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_2_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_2_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.2/m/m.0/cv2/act/Sigmoid
    Input: _model_2_m_m_0_cv2_conv_Conv_output_0 [1, 16, 56, 56]
    Output: _model_2_m_m_0_cv2_act_Sigmoid_output_0 [1, 16, 56, 56]
*/
void node__model_2_m_m_0_cv2_act_Sigmoid(const float _model_2_m_m_0_cv2_conv_Conv_output_0[1][16][56][56], float _model_2_m_m_0_cv2_act_Sigmoid_output_0[1][16][56][56]) {
    float *X_ptr = (float *)_model_2_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_2_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 50176; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.2/m/m.0/cv2/act/Mul
    Input: _model_2_m_m_0_cv2_conv_Conv_output_0 [1, 16, 56, 56]
    Input: _model_2_m_m_0_cv2_act_Sigmoid_output_0 [1, 16, 56, 56]
    Output: _model_2_m_m_0_cv2_act_Mul_output_0 [1, 16, 56, 56]
*/
void node__model_2_m_m_0_cv2_act_Mul(const float _model_2_m_m_0_cv2_conv_Conv_output_0[1][16][56][56], const float _model_2_m_m_0_cv2_act_Sigmoid_output_0[1][16][56][56], float _model_2_m_m_0_cv2_act_Mul_output_0[1][16][56][56]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 16; d1++) {
    for (int d2 = 0; d2 < 56; d2++) {
    for (int d3 = 0; d3 < 56; d3++) {
        _model_2_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_2_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_2_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.2/m/m.0/Add
    Input: _model_2_cv1_act_Mul_output_0 [1, 16, 56, 56]
    Input: _model_2_m_m_0_cv2_act_Mul_output_0 [1, 16, 56, 56]
    Output: _model_2_m_m_0_Add_output_0 [1, 16, 56, 56]
*/
void node__model_2_m_m_0_Add(const float _model_2_cv1_act_Mul_output_0[1][16][56][56], const float _model_2_m_m_0_cv2_act_Mul_output_0[1][16][56][56], float _model_2_m_m_0_Add_output_0[1][16][56][56]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 16; d1++) {
    for (int d2 = 0; d2 < 56; d2++) {
    for (int d3 = 0; d3 < 56; d3++) {
        _model_2_m_m_0_Add_output_0[d0][d1][d2][d3] = _model_2_cv1_act_Mul_output_0[d0][d1][d2][d3] + _model_2_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.2/Concat
    Input: _model_2_m_m_0_Add_output_0 [1, 16, 56, 56]
    Input: _model_2_cv2_act_Mul_output_0 [1, 16, 56, 56]
    Output: _model_2_Concat_output_0 [1, 32, 56, 56]
*/
void node__model_2_Concat(const float _model_2_m_m_0_Add_output_0[1][16][56][56], const float _model_2_cv2_act_Mul_output_0[1][16][56][56], float _model_2_Concat_output_0[1][32][56][56]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_2_m_m_0_Add_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 16; i1++) {
                for (int i2 = 0; i2 < 56; i2++) {
                    for (int i3 = 0; i3 < 56; i3++) {
                        _model_2_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_2_m_m_0_Add_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 16;
    // Copy tensor '_model_2_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 16; i1++) {
                for (int i2 = 0; i2 < 56; i2++) {
                    for (int i3 = 0; i3 < 56; i3++) {
                        _model_2_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_2_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 16;
}

/*  Operator: Conv 
    Name in model: /model.2/cv3/conv/Conv
    Input: _model_2_Concat_output_0 [1, 32, 56, 56]
    Input: model_2_cv3_conv_weight [32, 32, 1, 1]
    Input: model_2_cv3_conv_bias [32]
    Output: _model_2_cv3_conv_Conv_output_0 [1, 32, 56, 56]
*/
void node__model_2_cv3_conv_Conv(const float _model_2_Concat_output_0[1][32][56][56], const float model_2_cv3_conv_weight[32][32][1][1], const float model_2_cv3_conv_bias[32], float _model_2_cv3_conv_Conv_output_0[1][32][56][56]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 56, W_in = 56;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 56, W_out = 56;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_2_cv3_conv_bias[k];
                float ker_val = model_2_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_2_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_2_cv3_conv_Conv_output_0[n][k][h][w] += _model_2_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.2/cv3/act/Sigmoid
    Input: _model_2_cv3_conv_Conv_output_0 [1, 32, 56, 56]
    Output: _model_2_cv3_act_Sigmoid_output_0 [1, 32, 56, 56]
*/
void node__model_2_cv3_act_Sigmoid(const float _model_2_cv3_conv_Conv_output_0[1][32][56][56], float _model_2_cv3_act_Sigmoid_output_0[1][32][56][56]) {
    float *X_ptr = (float *)_model_2_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_2_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 100352; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.2/cv3/act/Mul
    Input: _model_2_cv3_conv_Conv_output_0 [1, 32, 56, 56]
    Input: _model_2_cv3_act_Sigmoid_output_0 [1, 32, 56, 56]
    Output: _model_2_cv3_act_Mul_output_0 [1, 32, 56, 56]
*/
void node__model_2_cv3_act_Mul(const float _model_2_cv3_conv_Conv_output_0[1][32][56][56], const float _model_2_cv3_act_Sigmoid_output_0[1][32][56][56], float _model_2_cv3_act_Mul_output_0[1][32][56][56]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 56; d2++) {
    for (int d3 = 0; d3 < 56; d3++) {
        _model_2_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_2_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_2_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.3/conv/Conv
    Input: _model_2_cv3_act_Mul_output_0 [1, 32, 56, 56]
    Input: model_3_conv_weight [64, 32, 3, 3]
    Input: model_3_conv_bias [64]
    Output: _model_3_conv_Conv_output_0 [1, 64, 28, 28]
*/
void node__model_3_conv_Conv(const float _model_2_cv3_act_Mul_output_0[1][32][56][56], const float model_3_conv_weight[64][32][3][3], const float model_3_conv_bias[64], float _model_3_conv_Conv_output_0[1][64][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 56, W_in = 56;
    const int K = 64, K_h = 3, K_w = 3;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 2, stride_w = 2;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_3_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_2_cv3_act_Mul_output_0[n][c][h_in][w_in] * model_3_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_3_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.3/act/Sigmoid
    Input: _model_3_conv_Conv_output_0 [1, 64, 28, 28]
    Output: _model_3_act_Sigmoid_output_0 [1, 64, 28, 28]
*/
void node__model_3_act_Sigmoid(const float _model_3_conv_Conv_output_0[1][64][28][28], float _model_3_act_Sigmoid_output_0[1][64][28][28]) {
    float *X_ptr = (float *)_model_3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_3_act_Sigmoid_output_0;
    for (int i = 0; i < 50176; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.3/act/Mul
    Input: _model_3_conv_Conv_output_0 [1, 64, 28, 28]
    Input: _model_3_act_Sigmoid_output_0 [1, 64, 28, 28]
    Output: _model_3_act_Mul_output_0 [1, 64, 28, 28]
*/
void node__model_3_act_Mul(const float _model_3_conv_Conv_output_0[1][64][28][28], const float _model_3_act_Sigmoid_output_0[1][64][28][28], float _model_3_act_Mul_output_0[1][64][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_3_act_Mul_output_0[d0][d1][d2][d3] = _model_3_conv_Conv_output_0[d0][d1][d2][d3] * _model_3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.4/cv1/conv/Conv
    Input: _model_3_act_Mul_output_0 [1, 64, 28, 28]
    Input: model_4_cv1_conv_weight [32, 64, 1, 1]
    Input: model_4_cv1_conv_bias [32]
    Output: _model_4_cv1_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_4_cv1_conv_Conv(const float _model_3_act_Mul_output_0[1][64][28][28], const float model_4_cv1_conv_weight[32][64][1][1], const float model_4_cv1_conv_bias[32], float _model_4_cv1_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 28, W_in = 28;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_4_cv1_conv_bias[k];
                float ker_val = model_4_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_4_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_4_cv1_conv_Conv_output_0[n][k][h][w] += _model_3_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.4/cv2/conv/Conv
    Input: _model_3_act_Mul_output_0 [1, 64, 28, 28]
    Input: model_4_cv2_conv_weight [32, 64, 1, 1]
    Input: model_4_cv2_conv_bias [32]
    Output: _model_4_cv2_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_4_cv2_conv_Conv(const float _model_3_act_Mul_output_0[1][64][28][28], const float model_4_cv2_conv_weight[32][64][1][1], const float model_4_cv2_conv_bias[32], float _model_4_cv2_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 28, W_in = 28;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_4_cv2_conv_bias[k];
                float ker_val = model_4_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_4_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_4_cv2_conv_Conv_output_0[n][k][h][w] += _model_3_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.4/cv1/act/Sigmoid
    Input: _model_4_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_4_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_4_cv1_act_Sigmoid(const float _model_4_cv1_conv_Conv_output_0[1][32][28][28], float _model_4_cv1_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_4_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_4_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.4/cv2/act/Sigmoid
    Input: _model_4_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_4_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_4_cv2_act_Sigmoid(const float _model_4_cv2_conv_Conv_output_0[1][32][28][28], float _model_4_cv2_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_4_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_4_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.4/cv1/act/Mul
    Input: _model_4_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_4_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_4_cv1_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_4_cv1_act_Mul(const float _model_4_cv1_conv_Conv_output_0[1][32][28][28], const float _model_4_cv1_act_Sigmoid_output_0[1][32][28][28], float _model_4_cv1_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_4_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_4_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.4/cv2/act/Mul
    Input: _model_4_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_4_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_4_cv2_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_4_cv2_act_Mul(const float _model_4_cv2_conv_Conv_output_0[1][32][28][28], const float _model_4_cv2_act_Sigmoid_output_0[1][32][28][28], float _model_4_cv2_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_4_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_4_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.4/m/m.0/cv1/conv/Conv
    Input: _model_4_cv1_act_Mul_output_0 [1, 32, 28, 28]
    Input: model_4_m_0_cv1_conv_weight [32, 32, 1, 1]
    Input: model_4_m_0_cv1_conv_bias [32]
    Output: _model_4_m_m_0_cv1_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_0_cv1_conv_Conv(const float _model_4_cv1_act_Mul_output_0[1][32][28][28], const float model_4_m_0_cv1_conv_weight[32][32][1][1], const float model_4_m_0_cv1_conv_bias[32], float _model_4_m_m_0_cv1_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 28, W_in = 28;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_4_m_0_cv1_conv_bias[k];
                float ker_val = model_4_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_4_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_4_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_4_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.4/m/m.0/cv1/act/Sigmoid
    Input: _model_4_m_m_0_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_0_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_0_cv1_act_Sigmoid(const float _model_4_m_m_0_cv1_conv_Conv_output_0[1][32][28][28], float _model_4_m_m_0_cv1_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_4_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_4_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.4/m/m.0/cv1/act/Mul
    Input: _model_4_m_m_0_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_4_m_m_0_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_0_cv1_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_0_cv1_act_Mul(const float _model_4_m_m_0_cv1_conv_Conv_output_0[1][32][28][28], const float _model_4_m_m_0_cv1_act_Sigmoid_output_0[1][32][28][28], float _model_4_m_m_0_cv1_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_4_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_4_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.4/m/m.0/cv2/conv/Conv
    Input: _model_4_m_m_0_cv1_act_Mul_output_0 [1, 32, 28, 28]
    Input: model_4_m_0_cv2_conv_weight [32, 32, 3, 3]
    Input: model_4_m_0_cv2_conv_bias [32]
    Output: _model_4_m_m_0_cv2_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_0_cv2_conv_Conv(const float _model_4_m_m_0_cv1_act_Mul_output_0[1][32][28][28], const float model_4_m_0_cv2_conv_weight[32][32][3][3], const float model_4_m_0_cv2_conv_bias[32], float _model_4_m_m_0_cv2_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 28, W_in = 28;
    const int K = 32, K_h = 3, K_w = 3;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_4_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_4_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_4_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_4_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.4/m/m.0/cv2/act/Sigmoid
    Input: _model_4_m_m_0_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_0_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_0_cv2_act_Sigmoid(const float _model_4_m_m_0_cv2_conv_Conv_output_0[1][32][28][28], float _model_4_m_m_0_cv2_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_4_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_4_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.4/m/m.0/cv2/act/Mul
    Input: _model_4_m_m_0_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_4_m_m_0_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_0_cv2_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_0_cv2_act_Mul(const float _model_4_m_m_0_cv2_conv_Conv_output_0[1][32][28][28], const float _model_4_m_m_0_cv2_act_Sigmoid_output_0[1][32][28][28], float _model_4_m_m_0_cv2_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_4_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_4_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.4/m/m.0/Add
    Input: _model_4_cv1_act_Mul_output_0 [1, 32, 28, 28]
    Input: _model_4_m_m_0_cv2_act_Mul_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_0_Add_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_0_Add(const float _model_4_cv1_act_Mul_output_0[1][32][28][28], const float _model_4_m_m_0_cv2_act_Mul_output_0[1][32][28][28], float _model_4_m_m_0_Add_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_m_m_0_Add_output_0[d0][d1][d2][d3] = _model_4_cv1_act_Mul_output_0[d0][d1][d2][d3] + _model_4_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.4/m/m.1/cv1/conv/Conv
    Input: _model_4_m_m_0_Add_output_0 [1, 32, 28, 28]
    Input: model_4_m_1_cv1_conv_weight [32, 32, 1, 1]
    Input: model_4_m_1_cv1_conv_bias [32]
    Output: _model_4_m_m_1_cv1_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_1_cv1_conv_Conv(const float _model_4_m_m_0_Add_output_0[1][32][28][28], const float model_4_m_1_cv1_conv_weight[32][32][1][1], const float model_4_m_1_cv1_conv_bias[32], float _model_4_m_m_1_cv1_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 28, W_in = 28;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_4_m_1_cv1_conv_bias[k];
                float ker_val = model_4_m_1_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_4_m_m_1_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_4_m_m_1_cv1_conv_Conv_output_0[n][k][h][w] += _model_4_m_m_0_Add_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.4/m/m.1/cv1/act/Sigmoid
    Input: _model_4_m_m_1_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_1_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_1_cv1_act_Sigmoid(const float _model_4_m_m_1_cv1_conv_Conv_output_0[1][32][28][28], float _model_4_m_m_1_cv1_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_4_m_m_1_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_4_m_m_1_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.4/m/m.1/cv1/act/Mul
    Input: _model_4_m_m_1_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_4_m_m_1_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_1_cv1_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_1_cv1_act_Mul(const float _model_4_m_m_1_cv1_conv_Conv_output_0[1][32][28][28], const float _model_4_m_m_1_cv1_act_Sigmoid_output_0[1][32][28][28], float _model_4_m_m_1_cv1_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_m_m_1_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_4_m_m_1_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_4_m_m_1_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.4/m/m.1/cv2/conv/Conv
    Input: _model_4_m_m_1_cv1_act_Mul_output_0 [1, 32, 28, 28]
    Input: model_4_m_1_cv2_conv_weight [32, 32, 3, 3]
    Input: model_4_m_1_cv2_conv_bias [32]
    Output: _model_4_m_m_1_cv2_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_1_cv2_conv_Conv(const float _model_4_m_m_1_cv1_act_Mul_output_0[1][32][28][28], const float model_4_m_1_cv2_conv_weight[32][32][3][3], const float model_4_m_1_cv2_conv_bias[32], float _model_4_m_m_1_cv2_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 28, W_in = 28;
    const int K = 32, K_h = 3, K_w = 3;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_4_m_1_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_4_m_m_1_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_4_m_1_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_4_m_m_1_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.4/m/m.1/cv2/act/Sigmoid
    Input: _model_4_m_m_1_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_1_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_1_cv2_act_Sigmoid(const float _model_4_m_m_1_cv2_conv_Conv_output_0[1][32][28][28], float _model_4_m_m_1_cv2_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_4_m_m_1_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_4_m_m_1_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.4/m/m.1/cv2/act/Mul
    Input: _model_4_m_m_1_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_4_m_m_1_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_1_cv2_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_1_cv2_act_Mul(const float _model_4_m_m_1_cv2_conv_Conv_output_0[1][32][28][28], const float _model_4_m_m_1_cv2_act_Sigmoid_output_0[1][32][28][28], float _model_4_m_m_1_cv2_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_m_m_1_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_4_m_m_1_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_4_m_m_1_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.4/m/m.1/Add
    Input: _model_4_m_m_0_Add_output_0 [1, 32, 28, 28]
    Input: _model_4_m_m_1_cv2_act_Mul_output_0 [1, 32, 28, 28]
    Output: _model_4_m_m_1_Add_output_0 [1, 32, 28, 28]
*/
void node__model_4_m_m_1_Add(const float _model_4_m_m_0_Add_output_0[1][32][28][28], const float _model_4_m_m_1_cv2_act_Mul_output_0[1][32][28][28], float _model_4_m_m_1_Add_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_m_m_1_Add_output_0[d0][d1][d2][d3] = _model_4_m_m_0_Add_output_0[d0][d1][d2][d3] + _model_4_m_m_1_cv2_act_Mul_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.4/Concat
    Input: _model_4_m_m_1_Add_output_0 [1, 32, 28, 28]
    Input: _model_4_cv2_act_Mul_output_0 [1, 32, 28, 28]
    Output: _model_4_Concat_output_0 [1, 64, 28, 28]
*/
void node__model_4_Concat(const float _model_4_m_m_1_Add_output_0[1][32][28][28], const float _model_4_cv2_act_Mul_output_0[1][32][28][28], float _model_4_Concat_output_0[1][64][28][28]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_4_m_m_1_Add_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 32; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        _model_4_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_4_m_m_1_Add_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 32;
    // Copy tensor '_model_4_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 32; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        _model_4_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_4_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 32;
}

/*  Operator: Conv 
    Name in model: /model.4/cv3/conv/Conv
    Input: _model_4_Concat_output_0 [1, 64, 28, 28]
    Input: model_4_cv3_conv_weight [64, 64, 1, 1]
    Input: model_4_cv3_conv_bias [64]
    Output: _model_4_cv3_conv_Conv_output_0 [1, 64, 28, 28]
*/
void node__model_4_cv3_conv_Conv(const float _model_4_Concat_output_0[1][64][28][28], const float model_4_cv3_conv_weight[64][64][1][1], const float model_4_cv3_conv_bias[64], float _model_4_cv3_conv_Conv_output_0[1][64][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 28, W_in = 28;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_4_cv3_conv_bias[k];
                float ker_val = model_4_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_4_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_4_cv3_conv_Conv_output_0[n][k][h][w] += _model_4_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.4/cv3/act/Sigmoid
    Input: _model_4_cv3_conv_Conv_output_0 [1, 64, 28, 28]
    Output: _model_4_cv3_act_Sigmoid_output_0 [1, 64, 28, 28]
*/
void node__model_4_cv3_act_Sigmoid(const float _model_4_cv3_conv_Conv_output_0[1][64][28][28], float _model_4_cv3_act_Sigmoid_output_0[1][64][28][28]) {
    float *X_ptr = (float *)_model_4_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_4_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 50176; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.4/cv3/act/Mul
    Input: _model_4_cv3_conv_Conv_output_0 [1, 64, 28, 28]
    Input: _model_4_cv3_act_Sigmoid_output_0 [1, 64, 28, 28]
    Output: _model_4_cv3_act_Mul_output_0 [1, 64, 28, 28]
*/
void node__model_4_cv3_act_Mul(const float _model_4_cv3_conv_Conv_output_0[1][64][28][28], const float _model_4_cv3_act_Sigmoid_output_0[1][64][28][28], float _model_4_cv3_act_Mul_output_0[1][64][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_4_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_4_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_4_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.5/conv/Conv
    Input: _model_4_cv3_act_Mul_output_0 [1, 64, 28, 28]
    Input: model_5_conv_weight [128, 64, 3, 3]
    Input: model_5_conv_bias [128]
    Output: _model_5_conv_Conv_output_0 [1, 128, 14, 14]
*/
void node__model_5_conv_Conv(const float _model_4_cv3_act_Mul_output_0[1][64][28][28], const float model_5_conv_weight[128][64][3][3], const float model_5_conv_bias[128], float _model_5_conv_Conv_output_0[1][128][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 28, W_in = 28;
    const int K = 128, K_h = 3, K_w = 3;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 2, stride_w = 2;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_5_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_4_cv3_act_Mul_output_0[n][c][h_in][w_in] * model_5_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_5_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.5/act/Sigmoid
    Input: _model_5_conv_Conv_output_0 [1, 128, 14, 14]
    Output: _model_5_act_Sigmoid_output_0 [1, 128, 14, 14]
*/
void node__model_5_act_Sigmoid(const float _model_5_conv_Conv_output_0[1][128][14][14], float _model_5_act_Sigmoid_output_0[1][128][14][14]) {
    float *X_ptr = (float *)_model_5_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_5_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.5/act/Mul
    Input: _model_5_conv_Conv_output_0 [1, 128, 14, 14]
    Input: _model_5_act_Sigmoid_output_0 [1, 128, 14, 14]
    Output: _model_5_act_Mul_output_0 [1, 128, 14, 14]
*/
void node__model_5_act_Mul(const float _model_5_conv_Conv_output_0[1][128][14][14], const float _model_5_act_Sigmoid_output_0[1][128][14][14], float _model_5_act_Mul_output_0[1][128][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_5_act_Mul_output_0[d0][d1][d2][d3] = _model_5_conv_Conv_output_0[d0][d1][d2][d3] * _model_5_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/cv1/conv/Conv
    Input: _model_5_act_Mul_output_0 [1, 128, 14, 14]
    Input: model_6_cv1_conv_weight [64, 128, 1, 1]
    Input: model_6_cv1_conv_bias [64]
    Output: _model_6_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_cv1_conv_Conv(const float _model_5_act_Mul_output_0[1][128][14][14], const float model_6_cv1_conv_weight[64][128][1][1], const float model_6_cv1_conv_bias[64], float _model_6_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_6_cv1_conv_bias[k];
                float ker_val = model_6_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_6_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_6_cv1_conv_Conv_output_0[n][k][h][w] += _model_5_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/cv2/conv/Conv
    Input: _model_5_act_Mul_output_0 [1, 128, 14, 14]
    Input: model_6_cv2_conv_weight [64, 128, 1, 1]
    Input: model_6_cv2_conv_bias [64]
    Output: _model_6_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_cv2_conv_Conv(const float _model_5_act_Mul_output_0[1][128][14][14], const float model_6_cv2_conv_weight[64][128][1][1], const float model_6_cv2_conv_bias[64], float _model_6_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_6_cv2_conv_bias[k];
                float ker_val = model_6_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_6_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_6_cv2_conv_Conv_output_0[n][k][h][w] += _model_5_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/cv1/act/Sigmoid
    Input: _model_6_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_cv1_act_Sigmoid(const float _model_6_cv1_conv_Conv_output_0[1][64][14][14], float _model_6_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/cv2/act/Sigmoid
    Input: _model_6_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_cv2_act_Sigmoid(const float _model_6_cv2_conv_Conv_output_0[1][64][14][14], float _model_6_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/cv1/act/Mul
    Input: _model_6_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_cv1_act_Mul(const float _model_6_cv1_conv_Conv_output_0[1][64][14][14], const float _model_6_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_6_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_6_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.6/cv2/act/Mul
    Input: _model_6_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_cv2_act_Mul(const float _model_6_cv2_conv_Conv_output_0[1][64][14][14], const float _model_6_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_6_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_6_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/m/m.0/cv1/conv/Conv
    Input: _model_6_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_6_m_0_cv1_conv_weight [64, 64, 1, 1]
    Input: model_6_m_0_cv1_conv_bias [64]
    Output: _model_6_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_0_cv1_conv_Conv(const float _model_6_cv1_act_Mul_output_0[1][64][14][14], const float model_6_m_0_cv1_conv_weight[64][64][1][1], const float model_6_m_0_cv1_conv_bias[64], float _model_6_m_m_0_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_6_m_0_cv1_conv_bias[k];
                float ker_val = model_6_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_6_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_6_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_6_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/m/m.0/cv1/act/Sigmoid
    Input: _model_6_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_0_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_0_cv1_act_Sigmoid(const float _model_6_m_m_0_cv1_conv_Conv_output_0[1][64][14][14], float _model_6_m_m_0_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/m/m.0/cv1/act/Mul
    Input: _model_6_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_0_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_0_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_0_cv1_act_Mul(const float _model_6_m_m_0_cv1_conv_Conv_output_0[1][64][14][14], const float _model_6_m_m_0_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_6_m_m_0_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_6_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/m/m.0/cv2/conv/Conv
    Input: _model_6_m_m_0_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_6_m_0_cv2_conv_weight [64, 64, 3, 3]
    Input: model_6_m_0_cv2_conv_bias [64]
    Output: _model_6_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_0_cv2_conv_Conv(const float _model_6_m_m_0_cv1_act_Mul_output_0[1][64][14][14], const float model_6_m_0_cv2_conv_weight[64][64][3][3], const float model_6_m_0_cv2_conv_bias[64], float _model_6_m_m_0_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 3, K_w = 3;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_6_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_6_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_6_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_6_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/m/m.0/cv2/act/Sigmoid
    Input: _model_6_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_0_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_0_cv2_act_Sigmoid(const float _model_6_m_m_0_cv2_conv_Conv_output_0[1][64][14][14], float _model_6_m_m_0_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/m/m.0/cv2/act/Mul
    Input: _model_6_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_0_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_0_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_0_cv2_act_Mul(const float _model_6_m_m_0_cv2_conv_Conv_output_0[1][64][14][14], const float _model_6_m_m_0_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_6_m_m_0_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_6_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.6/m/m.0/Add
    Input: _model_6_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_0_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_0_Add_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_0_Add(const float _model_6_cv1_act_Mul_output_0[1][64][14][14], const float _model_6_m_m_0_cv2_act_Mul_output_0[1][64][14][14], float _model_6_m_m_0_Add_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_0_Add_output_0[d0][d1][d2][d3] = _model_6_cv1_act_Mul_output_0[d0][d1][d2][d3] + _model_6_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/m/m.1/cv1/conv/Conv
    Input: _model_6_m_m_0_Add_output_0 [1, 64, 14, 14]
    Input: model_6_m_1_cv1_conv_weight [64, 64, 1, 1]
    Input: model_6_m_1_cv1_conv_bias [64]
    Output: _model_6_m_m_1_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_1_cv1_conv_Conv(const float _model_6_m_m_0_Add_output_0[1][64][14][14], const float model_6_m_1_cv1_conv_weight[64][64][1][1], const float model_6_m_1_cv1_conv_bias[64], float _model_6_m_m_1_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_6_m_1_cv1_conv_bias[k];
                float ker_val = model_6_m_1_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_6_m_m_1_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_6_m_m_1_cv1_conv_Conv_output_0[n][k][h][w] += _model_6_m_m_0_Add_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/m/m.1/cv1/act/Sigmoid
    Input: _model_6_m_m_1_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_1_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_1_cv1_act_Sigmoid(const float _model_6_m_m_1_cv1_conv_Conv_output_0[1][64][14][14], float _model_6_m_m_1_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_m_m_1_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_m_m_1_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/m/m.1/cv1/act/Mul
    Input: _model_6_m_m_1_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_1_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_1_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_1_cv1_act_Mul(const float _model_6_m_m_1_cv1_conv_Conv_output_0[1][64][14][14], const float _model_6_m_m_1_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_6_m_m_1_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_1_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_6_m_m_1_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_m_m_1_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/m/m.1/cv2/conv/Conv
    Input: _model_6_m_m_1_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_6_m_1_cv2_conv_weight [64, 64, 3, 3]
    Input: model_6_m_1_cv2_conv_bias [64]
    Output: _model_6_m_m_1_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_1_cv2_conv_Conv(const float _model_6_m_m_1_cv1_act_Mul_output_0[1][64][14][14], const float model_6_m_1_cv2_conv_weight[64][64][3][3], const float model_6_m_1_cv2_conv_bias[64], float _model_6_m_m_1_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 3, K_w = 3;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_6_m_1_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_6_m_m_1_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_6_m_1_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_6_m_m_1_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/m/m.1/cv2/act/Sigmoid
    Input: _model_6_m_m_1_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_1_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_1_cv2_act_Sigmoid(const float _model_6_m_m_1_cv2_conv_Conv_output_0[1][64][14][14], float _model_6_m_m_1_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_m_m_1_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_m_m_1_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/m/m.1/cv2/act/Mul
    Input: _model_6_m_m_1_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_1_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_1_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_1_cv2_act_Mul(const float _model_6_m_m_1_cv2_conv_Conv_output_0[1][64][14][14], const float _model_6_m_m_1_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_6_m_m_1_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_1_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_6_m_m_1_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_m_m_1_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.6/m/m.1/Add
    Input: _model_6_m_m_0_Add_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_1_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_1_Add_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_1_Add(const float _model_6_m_m_0_Add_output_0[1][64][14][14], const float _model_6_m_m_1_cv2_act_Mul_output_0[1][64][14][14], float _model_6_m_m_1_Add_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_1_Add_output_0[d0][d1][d2][d3] = _model_6_m_m_0_Add_output_0[d0][d1][d2][d3] + _model_6_m_m_1_cv2_act_Mul_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/m/m.2/cv1/conv/Conv
    Input: _model_6_m_m_1_Add_output_0 [1, 64, 14, 14]
    Input: model_6_m_2_cv1_conv_weight [64, 64, 1, 1]
    Input: model_6_m_2_cv1_conv_bias [64]
    Output: _model_6_m_m_2_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_2_cv1_conv_Conv(const float _model_6_m_m_1_Add_output_0[1][64][14][14], const float model_6_m_2_cv1_conv_weight[64][64][1][1], const float model_6_m_2_cv1_conv_bias[64], float _model_6_m_m_2_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_6_m_2_cv1_conv_bias[k];
                float ker_val = model_6_m_2_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_6_m_m_2_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_6_m_m_2_cv1_conv_Conv_output_0[n][k][h][w] += _model_6_m_m_1_Add_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/m/m.2/cv1/act/Sigmoid
    Input: _model_6_m_m_2_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_2_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_2_cv1_act_Sigmoid(const float _model_6_m_m_2_cv1_conv_Conv_output_0[1][64][14][14], float _model_6_m_m_2_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_m_m_2_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_m_m_2_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/m/m.2/cv1/act/Mul
    Input: _model_6_m_m_2_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_2_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_2_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_2_cv1_act_Mul(const float _model_6_m_m_2_cv1_conv_Conv_output_0[1][64][14][14], const float _model_6_m_m_2_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_6_m_m_2_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_2_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_6_m_m_2_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_m_m_2_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.6/m/m.2/cv2/conv/Conv
    Input: _model_6_m_m_2_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_6_m_2_cv2_conv_weight [64, 64, 3, 3]
    Input: model_6_m_2_cv2_conv_bias [64]
    Output: _model_6_m_m_2_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_2_cv2_conv_Conv(const float _model_6_m_m_2_cv1_act_Mul_output_0[1][64][14][14], const float model_6_m_2_cv2_conv_weight[64][64][3][3], const float model_6_m_2_cv2_conv_bias[64], float _model_6_m_m_2_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 3, K_w = 3;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_6_m_2_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_6_m_m_2_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_6_m_2_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_6_m_m_2_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/m/m.2/cv2/act/Sigmoid
    Input: _model_6_m_m_2_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_2_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_2_cv2_act_Sigmoid(const float _model_6_m_m_2_cv2_conv_Conv_output_0[1][64][14][14], float _model_6_m_m_2_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_6_m_m_2_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_m_m_2_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/m/m.2/cv2/act/Mul
    Input: _model_6_m_m_2_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_2_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_2_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_2_cv2_act_Mul(const float _model_6_m_m_2_cv2_conv_Conv_output_0[1][64][14][14], const float _model_6_m_m_2_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_6_m_m_2_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_2_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_6_m_m_2_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_m_m_2_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.6/m/m.2/Add
    Input: _model_6_m_m_1_Add_output_0 [1, 64, 14, 14]
    Input: _model_6_m_m_2_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Output: _model_6_m_m_2_Add_output_0 [1, 64, 14, 14]
*/
void node__model_6_m_m_2_Add(const float _model_6_m_m_1_Add_output_0[1][64][14][14], const float _model_6_m_m_2_cv2_act_Mul_output_0[1][64][14][14], float _model_6_m_m_2_Add_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_m_m_2_Add_output_0[d0][d1][d2][d3] = _model_6_m_m_1_Add_output_0[d0][d1][d2][d3] + _model_6_m_m_2_cv2_act_Mul_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.6/Concat
    Input: _model_6_m_m_2_Add_output_0 [1, 64, 14, 14]
    Input: _model_6_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Output: _model_6_Concat_output_0 [1, 128, 14, 14]
*/
void node__model_6_Concat(const float _model_6_m_m_2_Add_output_0[1][64][14][14], const float _model_6_cv2_act_Mul_output_0[1][64][14][14], float _model_6_Concat_output_0[1][128][14][14]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_6_m_m_2_Add_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_6_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_6_m_m_2_Add_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
    // Copy tensor '_model_6_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_6_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_6_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
}

/*  Operator: Conv 
    Name in model: /model.6/cv3/conv/Conv
    Input: _model_6_Concat_output_0 [1, 128, 14, 14]
    Input: model_6_cv3_conv_weight [128, 128, 1, 1]
    Input: model_6_cv3_conv_bias [128]
    Output: _model_6_cv3_conv_Conv_output_0 [1, 128, 14, 14]
*/
void node__model_6_cv3_conv_Conv(const float _model_6_Concat_output_0[1][128][14][14], const float model_6_cv3_conv_weight[128][128][1][1], const float model_6_cv3_conv_bias[128], float _model_6_cv3_conv_Conv_output_0[1][128][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_6_cv3_conv_bias[k];
                float ker_val = model_6_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_6_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_6_cv3_conv_Conv_output_0[n][k][h][w] += _model_6_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.6/cv3/act/Sigmoid
    Input: _model_6_cv3_conv_Conv_output_0 [1, 128, 14, 14]
    Output: _model_6_cv3_act_Sigmoid_output_0 [1, 128, 14, 14]
*/
void node__model_6_cv3_act_Sigmoid(const float _model_6_cv3_conv_Conv_output_0[1][128][14][14], float _model_6_cv3_act_Sigmoid_output_0[1][128][14][14]) {
    float *X_ptr = (float *)_model_6_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_6_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.6/cv3/act/Mul
    Input: _model_6_cv3_conv_Conv_output_0 [1, 128, 14, 14]
    Input: _model_6_cv3_act_Sigmoid_output_0 [1, 128, 14, 14]
    Output: _model_6_cv3_act_Mul_output_0 [1, 128, 14, 14]
*/
void node__model_6_cv3_act_Mul(const float _model_6_cv3_conv_Conv_output_0[1][128][14][14], const float _model_6_cv3_act_Sigmoid_output_0[1][128][14][14], float _model_6_cv3_act_Mul_output_0[1][128][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_6_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_6_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_6_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.7/conv/Conv
    Input: _model_6_cv3_act_Mul_output_0 [1, 128, 14, 14]
    Input: model_7_conv_weight [256, 128, 3, 3]
    Input: model_7_conv_bias [256]
    Output: _model_7_conv_Conv_output_0 [1, 256, 7, 7]
*/
void node__model_7_conv_Conv(const float _model_6_cv3_act_Mul_output_0[1][128][14][14], const float model_7_conv_weight[256][128][3][3], const float model_7_conv_bias[256], float _model_7_conv_Conv_output_0[1][256][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 256, K_h = 3, K_w = 3;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 2, stride_w = 2;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_7_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_6_cv3_act_Mul_output_0[n][c][h_in][w_in] * model_7_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_7_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.7/act/Sigmoid
    Input: _model_7_conv_Conv_output_0 [1, 256, 7, 7]
    Output: _model_7_act_Sigmoid_output_0 [1, 256, 7, 7]
*/
void node__model_7_act_Sigmoid(const float _model_7_conv_Conv_output_0[1][256][7][7], float _model_7_act_Sigmoid_output_0[1][256][7][7]) {
    float *X_ptr = (float *)_model_7_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_7_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.7/act/Mul
    Input: _model_7_conv_Conv_output_0 [1, 256, 7, 7]
    Input: _model_7_act_Sigmoid_output_0 [1, 256, 7, 7]
    Output: _model_7_act_Mul_output_0 [1, 256, 7, 7]
*/
void node__model_7_act_Mul(const float _model_7_conv_Conv_output_0[1][256][7][7], const float _model_7_act_Sigmoid_output_0[1][256][7][7], float _model_7_act_Mul_output_0[1][256][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 256; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_7_act_Mul_output_0[d0][d1][d2][d3] = _model_7_conv_Conv_output_0[d0][d1][d2][d3] * _model_7_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.8/cv1/conv/Conv
    Input: _model_7_act_Mul_output_0 [1, 256, 7, 7]
    Input: model_8_cv1_conv_weight [128, 256, 1, 1]
    Input: model_8_cv1_conv_bias [128]
    Output: _model_8_cv1_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_8_cv1_conv_Conv(const float _model_7_act_Mul_output_0[1][256][7][7], const float model_8_cv1_conv_weight[128][256][1][1], const float model_8_cv1_conv_bias[128], float _model_8_cv1_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_8_cv1_conv_bias[k];
                float ker_val = model_8_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_8_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_8_cv1_conv_Conv_output_0[n][k][h][w] += _model_7_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.8/cv2/conv/Conv
    Input: _model_7_act_Mul_output_0 [1, 256, 7, 7]
    Input: model_8_cv2_conv_weight [128, 256, 1, 1]
    Input: model_8_cv2_conv_bias [128]
    Output: _model_8_cv2_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_8_cv2_conv_Conv(const float _model_7_act_Mul_output_0[1][256][7][7], const float model_8_cv2_conv_weight[128][256][1][1], const float model_8_cv2_conv_bias[128], float _model_8_cv2_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_8_cv2_conv_bias[k];
                float ker_val = model_8_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_8_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_8_cv2_conv_Conv_output_0[n][k][h][w] += _model_7_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.8/cv1/act/Sigmoid
    Input: _model_8_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_8_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_8_cv1_act_Sigmoid(const float _model_8_cv1_conv_Conv_output_0[1][128][7][7], float _model_8_cv1_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_8_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_8_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.8/cv2/act/Sigmoid
    Input: _model_8_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_8_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_8_cv2_act_Sigmoid(const float _model_8_cv2_conv_Conv_output_0[1][128][7][7], float _model_8_cv2_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_8_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_8_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.8/cv1/act/Mul
    Input: _model_8_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_8_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_8_cv1_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_8_cv1_act_Mul(const float _model_8_cv1_conv_Conv_output_0[1][128][7][7], const float _model_8_cv1_act_Sigmoid_output_0[1][128][7][7], float _model_8_cv1_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_8_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_8_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_8_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.8/cv2/act/Mul
    Input: _model_8_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_8_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_8_cv2_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_8_cv2_act_Mul(const float _model_8_cv2_conv_Conv_output_0[1][128][7][7], const float _model_8_cv2_act_Sigmoid_output_0[1][128][7][7], float _model_8_cv2_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_8_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_8_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_8_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.8/m/m.0/cv1/conv/Conv
    Input: _model_8_cv1_act_Mul_output_0 [1, 128, 7, 7]
    Input: model_8_m_0_cv1_conv_weight [128, 128, 1, 1]
    Input: model_8_m_0_cv1_conv_bias [128]
    Output: _model_8_m_m_0_cv1_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_8_m_m_0_cv1_conv_Conv(const float _model_8_cv1_act_Mul_output_0[1][128][7][7], const float model_8_m_0_cv1_conv_weight[128][128][1][1], const float model_8_m_0_cv1_conv_bias[128], float _model_8_m_m_0_cv1_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_8_m_0_cv1_conv_bias[k];
                float ker_val = model_8_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_8_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_8_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_8_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.8/m/m.0/cv1/act/Sigmoid
    Input: _model_8_m_m_0_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_8_m_m_0_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_8_m_m_0_cv1_act_Sigmoid(const float _model_8_m_m_0_cv1_conv_Conv_output_0[1][128][7][7], float _model_8_m_m_0_cv1_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_8_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_8_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.8/m/m.0/cv1/act/Mul
    Input: _model_8_m_m_0_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_8_m_m_0_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_8_m_m_0_cv1_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_8_m_m_0_cv1_act_Mul(const float _model_8_m_m_0_cv1_conv_Conv_output_0[1][128][7][7], const float _model_8_m_m_0_cv1_act_Sigmoid_output_0[1][128][7][7], float _model_8_m_m_0_cv1_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_8_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_8_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_8_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.8/m/m.0/cv2/conv/Conv
    Input: _model_8_m_m_0_cv1_act_Mul_output_0 [1, 128, 7, 7]
    Input: model_8_m_0_cv2_conv_weight [128, 128, 3, 3]
    Input: model_8_m_0_cv2_conv_bias [128]
    Output: _model_8_m_m_0_cv2_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_8_m_m_0_cv2_conv_Conv(const float _model_8_m_m_0_cv1_act_Mul_output_0[1][128][7][7], const float model_8_m_0_cv2_conv_weight[128][128][3][3], const float model_8_m_0_cv2_conv_bias[128], float _model_8_m_m_0_cv2_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 7, W_in = 7;
    const int K = 128, K_h = 3, K_w = 3;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_8_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_8_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_8_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_8_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.8/m/m.0/cv2/act/Sigmoid
    Input: _model_8_m_m_0_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_8_m_m_0_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_8_m_m_0_cv2_act_Sigmoid(const float _model_8_m_m_0_cv2_conv_Conv_output_0[1][128][7][7], float _model_8_m_m_0_cv2_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_8_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_8_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.8/m/m.0/cv2/act/Mul
    Input: _model_8_m_m_0_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_8_m_m_0_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_8_m_m_0_cv2_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_8_m_m_0_cv2_act_Mul(const float _model_8_m_m_0_cv2_conv_Conv_output_0[1][128][7][7], const float _model_8_m_m_0_cv2_act_Sigmoid_output_0[1][128][7][7], float _model_8_m_m_0_cv2_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_8_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_8_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_8_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.8/m/m.0/Add
    Input: _model_8_cv1_act_Mul_output_0 [1, 128, 7, 7]
    Input: _model_8_m_m_0_cv2_act_Mul_output_0 [1, 128, 7, 7]
    Output: _model_8_m_m_0_Add_output_0 [1, 128, 7, 7]
*/
void node__model_8_m_m_0_Add(const float _model_8_cv1_act_Mul_output_0[1][128][7][7], const float _model_8_m_m_0_cv2_act_Mul_output_0[1][128][7][7], float _model_8_m_m_0_Add_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_8_m_m_0_Add_output_0[d0][d1][d2][d3] = _model_8_cv1_act_Mul_output_0[d0][d1][d2][d3] + _model_8_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.8/Concat
    Input: _model_8_m_m_0_Add_output_0 [1, 128, 7, 7]
    Input: _model_8_cv2_act_Mul_output_0 [1, 128, 7, 7]
    Output: _model_8_Concat_output_0 [1, 256, 7, 7]
*/
void node__model_8_Concat(const float _model_8_m_m_0_Add_output_0[1][128][7][7], const float _model_8_cv2_act_Mul_output_0[1][128][7][7], float _model_8_Concat_output_0[1][256][7][7]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_8_m_m_0_Add_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_8_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_8_m_m_0_Add_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
    // Copy tensor '_model_8_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_8_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_8_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
}

/*  Operator: Conv 
    Name in model: /model.8/cv3/conv/Conv
    Input: _model_8_Concat_output_0 [1, 256, 7, 7]
    Input: model_8_cv3_conv_weight [256, 256, 1, 1]
    Input: model_8_cv3_conv_bias [256]
    Output: _model_8_cv3_conv_Conv_output_0 [1, 256, 7, 7]
*/
void node__model_8_cv3_conv_Conv(const float _model_8_Concat_output_0[1][256][7][7], const float model_8_cv3_conv_weight[256][256][1][1], const float model_8_cv3_conv_bias[256], float _model_8_cv3_conv_Conv_output_0[1][256][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 256, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_8_cv3_conv_bias[k];
                float ker_val = model_8_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_8_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_8_cv3_conv_Conv_output_0[n][k][h][w] += _model_8_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.8/cv3/act/Sigmoid
    Input: _model_8_cv3_conv_Conv_output_0 [1, 256, 7, 7]
    Output: _model_8_cv3_act_Sigmoid_output_0 [1, 256, 7, 7]
*/
void node__model_8_cv3_act_Sigmoid(const float _model_8_cv3_conv_Conv_output_0[1][256][7][7], float _model_8_cv3_act_Sigmoid_output_0[1][256][7][7]) {
    float *X_ptr = (float *)_model_8_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_8_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.8/cv3/act/Mul
    Input: _model_8_cv3_conv_Conv_output_0 [1, 256, 7, 7]
    Input: _model_8_cv3_act_Sigmoid_output_0 [1, 256, 7, 7]
    Output: _model_8_cv3_act_Mul_output_0 [1, 256, 7, 7]
*/
void node__model_8_cv3_act_Mul(const float _model_8_cv3_conv_Conv_output_0[1][256][7][7], const float _model_8_cv3_act_Sigmoid_output_0[1][256][7][7], float _model_8_cv3_act_Mul_output_0[1][256][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 256; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_8_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_8_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_8_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.9/cv1/conv/Conv
    Input: _model_8_cv3_act_Mul_output_0 [1, 256, 7, 7]
    Input: model_9_cv1_conv_weight [128, 256, 1, 1]
    Input: model_9_cv1_conv_bias [128]
    Output: _model_9_cv1_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_9_cv1_conv_Conv(const float _model_8_cv3_act_Mul_output_0[1][256][7][7], const float model_9_cv1_conv_weight[128][256][1][1], const float model_9_cv1_conv_bias[128], float _model_9_cv1_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_9_cv1_conv_bias[k];
                float ker_val = model_9_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_9_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_9_cv1_conv_Conv_output_0[n][k][h][w] += _model_8_cv3_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.9/cv1/act/Sigmoid
    Input: _model_9_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_9_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_9_cv1_act_Sigmoid(const float _model_9_cv1_conv_Conv_output_0[1][128][7][7], float _model_9_cv1_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_9_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_9_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.9/cv1/act/Mul
    Input: _model_9_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_9_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_9_cv1_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_9_cv1_act_Mul(const float _model_9_cv1_conv_Conv_output_0[1][128][7][7], const float _model_9_cv1_act_Sigmoid_output_0[1][128][7][7], float _model_9_cv1_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_9_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_9_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_9_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: MaxPool 
    Name in model: /model.9/m/MaxPool
    Input: _model_9_cv1_act_Mul_output_0 [1, 128, 7, 7]
    Output: _model_9_m_MaxPool_output_0 [1, 128, 7, 7]
*/
    /* MaxPool
     * ceil_mode: 0
     * dilations: 1 1
     * kernel_shape: 5 5
     * pads: 2 2 2 2
     * strides: 1 1
     */
void node__model_9_m_MaxPool(const float _model_9_cv1_act_Mul_output_0[1][128][7][7], float _model_9_m_MaxPool_output_0[1][128][7][7]) {
    const int batch = 1, channels = 128;
    const int in_h = 7, in_w = 7;
    const int out_h = 7, out_w = 7;
    const int kernel_h = 5, kernel_w = 5;
    const int stride_h = 1, stride_w = 1;
    const int pad_begin_h = 2, pad_end_h = 2, pad_begin_w = 2, pad_end_w = 2;

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -INFINITY;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int h_in = oh * stride_h - pad_begin_h + kh;
                            int w_in = ow * stride_w - pad_begin_w + kw;
                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                                float val = _model_9_cv1_act_Mul_output_0[n][c][h_in][w_in];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    _model_9_m_MaxPool_output_0[n][c][oh][ow] = max_val;
                }
            }
        }
    }
}

/*  Operator: MaxPool 
    Name in model: /model.9/m_1/MaxPool
    Input: _model_9_m_MaxPool_output_0 [1, 128, 7, 7]
    Output: _model_9_m_1_MaxPool_output_0 [1, 128, 7, 7]
*/
    /* MaxPool
     * ceil_mode: 0
     * dilations: 1 1
     * kernel_shape: 5 5
     * pads: 2 2 2 2
     * strides: 1 1
     */
void node__model_9_m_1_MaxPool(const float _model_9_m_MaxPool_output_0[1][128][7][7], float _model_9_m_1_MaxPool_output_0[1][128][7][7]) {
    const int batch = 1, channels = 128;
    const int in_h = 7, in_w = 7;
    const int out_h = 7, out_w = 7;
    const int kernel_h = 5, kernel_w = 5;
    const int stride_h = 1, stride_w = 1;
    const int pad_begin_h = 2, pad_end_h = 2, pad_begin_w = 2, pad_end_w = 2;

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -INFINITY;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int h_in = oh * stride_h - pad_begin_h + kh;
                            int w_in = ow * stride_w - pad_begin_w + kw;
                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                                float val = _model_9_m_MaxPool_output_0[n][c][h_in][w_in];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    _model_9_m_1_MaxPool_output_0[n][c][oh][ow] = max_val;
                }
            }
        }
    }
}

/*  Operator: MaxPool 
    Name in model: /model.9/m_2/MaxPool
    Input: _model_9_m_1_MaxPool_output_0 [1, 128, 7, 7]
    Output: _model_9_m_2_MaxPool_output_0 [1, 128, 7, 7]
*/
    /* MaxPool
     * ceil_mode: 0
     * dilations: 1 1
     * kernel_shape: 5 5
     * pads: 2 2 2 2
     * strides: 1 1
     */
void node__model_9_m_2_MaxPool(const float _model_9_m_1_MaxPool_output_0[1][128][7][7], float _model_9_m_2_MaxPool_output_0[1][128][7][7]) {
    const int batch = 1, channels = 128;
    const int in_h = 7, in_w = 7;
    const int out_h = 7, out_w = 7;
    const int kernel_h = 5, kernel_w = 5;
    const int stride_h = 1, stride_w = 1;
    const int pad_begin_h = 2, pad_end_h = 2, pad_begin_w = 2, pad_end_w = 2;

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -INFINITY;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int h_in = oh * stride_h - pad_begin_h + kh;
                            int w_in = ow * stride_w - pad_begin_w + kw;
                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                                float val = _model_9_m_1_MaxPool_output_0[n][c][h_in][w_in];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    _model_9_m_2_MaxPool_output_0[n][c][oh][ow] = max_val;
                }
            }
        }
    }
}

/*  Operator: Concat 
    Name in model: /model.9/Concat
    Input: _model_9_cv1_act_Mul_output_0 [1, 128, 7, 7]
    Input: _model_9_m_MaxPool_output_0 [1, 128, 7, 7]
    Input: _model_9_m_1_MaxPool_output_0 [1, 128, 7, 7]
    Input: _model_9_m_2_MaxPool_output_0 [1, 128, 7, 7]
    Output: _model_9_Concat_output_0 [1, 512, 7, 7]
*/
void node__model_9_Concat(const float _model_9_cv1_act_Mul_output_0[1][128][7][7], const float _model_9_m_MaxPool_output_0[1][128][7][7], const float _model_9_m_1_MaxPool_output_0[1][128][7][7], const float _model_9_m_2_MaxPool_output_0[1][128][7][7], float _model_9_Concat_output_0[1][512][7][7]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_9_cv1_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_9_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_9_cv1_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
    // Copy tensor '_model_9_m_MaxPool_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_9_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_9_m_MaxPool_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
    // Copy tensor '_model_9_m_1_MaxPool_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_9_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_9_m_1_MaxPool_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
    // Copy tensor '_model_9_m_2_MaxPool_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_9_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_9_m_2_MaxPool_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
}

/*  Operator: Conv 
    Name in model: /model.9/cv2/conv/Conv
    Input: _model_9_Concat_output_0 [1, 512, 7, 7]
    Input: model_9_cv2_conv_weight [256, 512, 1, 1]
    Input: model_9_cv2_conv_bias [256]
    Output: _model_9_cv2_conv_Conv_output_0 [1, 256, 7, 7]
*/
void node__model_9_cv2_conv_Conv(const float _model_9_Concat_output_0[1][512][7][7], const float model_9_cv2_conv_weight[256][512][1][1], const float model_9_cv2_conv_bias[256], float _model_9_cv2_conv_Conv_output_0[1][256][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 512, H_in = 7, W_in = 7;
    const int K = 256, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_9_cv2_conv_bias[k];
                float ker_val = model_9_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_9_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_9_cv2_conv_Conv_output_0[n][k][h][w] += _model_9_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.9/cv2/act/Sigmoid
    Input: _model_9_cv2_conv_Conv_output_0 [1, 256, 7, 7]
    Output: _model_9_cv2_act_Sigmoid_output_0 [1, 256, 7, 7]
*/
void node__model_9_cv2_act_Sigmoid(const float _model_9_cv2_conv_Conv_output_0[1][256][7][7], float _model_9_cv2_act_Sigmoid_output_0[1][256][7][7]) {
    float *X_ptr = (float *)_model_9_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_9_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.9/cv2/act/Mul
    Input: _model_9_cv2_conv_Conv_output_0 [1, 256, 7, 7]
    Input: _model_9_cv2_act_Sigmoid_output_0 [1, 256, 7, 7]
    Output: _model_9_cv2_act_Mul_output_0 [1, 256, 7, 7]
*/
void node__model_9_cv2_act_Mul(const float _model_9_cv2_conv_Conv_output_0[1][256][7][7], const float _model_9_cv2_act_Sigmoid_output_0[1][256][7][7], float _model_9_cv2_act_Mul_output_0[1][256][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 256; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_9_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_9_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_9_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.10/conv/Conv
    Input: _model_9_cv2_act_Mul_output_0 [1, 256, 7, 7]
    Input: model_10_conv_weight [128, 256, 1, 1]
    Input: model_10_conv_bias [128]
    Output: _model_10_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_10_conv_Conv(const float _model_9_cv2_act_Mul_output_0[1][256][7][7], const float model_10_conv_weight[128][256][1][1], const float model_10_conv_bias[128], float _model_10_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_10_conv_bias[k];
                float ker_val = model_10_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_10_conv_Conv_output_0[n][k][h][w] = b;
                        _model_10_conv_Conv_output_0[n][k][h][w] += _model_9_cv2_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.10/act/Sigmoid
    Input: _model_10_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_10_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_10_act_Sigmoid(const float _model_10_conv_Conv_output_0[1][128][7][7], float _model_10_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_10_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_10_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.10/act/Mul
    Input: _model_10_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_10_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_10_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_10_act_Mul(const float _model_10_conv_Conv_output_0[1][128][7][7], const float _model_10_act_Sigmoid_output_0[1][128][7][7], float _model_10_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_10_act_Mul_output_0[d0][d1][d2][d3] = _model_10_conv_Conv_output_0[d0][d1][d2][d3] * _model_10_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Resize 
    Name in model: /model.11/Resize
    Input: _model_10_act_Mul_output_0 [1, 128, 7, 7]
    Input:  None
    Input:  None
    Input: _model_11_Concat_1_output_0 [4]
    Output: _model_11_Resize_output_0 [1, 128, 14, 14]
*/
void node__model_11_Resize(const float _model_10_act_Mul_output_0[1][128][7][7], const int64_t _model_11_Concat_1_output_0[4], float _model_11_Resize_output_0[1][128][14][14]) {
    /*Resize: mode=b'nearest', coord_transform=b'asymmetric', cubic_a=-0.75, exclude_outside=0, extrapolation=0.0, nearest_mode=b'floor'*/
    const int N = 1;
    const int C = 128;
    const int H_in = 7;
    const int W_in = 7;
    const int H_out = _model_11_Concat_1_output_0[2]; // 14;
    const int W_out = _model_11_Concat_1_output_0[3]; // 14;

    float scale_h = (float)H_out / H_in;
    float scale_w = (float)W_out / W_in;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < H_out; ++y) {
                for (int x = 0; x < W_out; ++x) {
                    float y_in = ((float)y + 0.5f) / scale_h - 0.5f;
                    float x_in = ((float)x + 0.5f) / scale_w - 0.5f;
                    int y_index = (int)floorf(y_in);
                    int x_index = (int)floorf(x_in);
                    y_index = y_index < 0 ? 0 : (y_index >= H_in ? H_in - 1 : y_index);
                    x_index = x_index < 0 ? 0 : (x_index >= W_in ? W_in - 1 : x_index);
                    _model_11_Resize_output_0[n][c][y][x] = _model_10_act_Mul_output_0[n][c][y_index][x_index];
                }
            }
        }
    }
}

/*  Operator: Concat 
    Name in model: /model.12/Concat
    Input: _model_11_Resize_output_0 [1, 128, 14, 14]
    Input: _model_6_cv3_act_Mul_output_0 [1, 128, 14, 14]
    Output: _model_12_Concat_output_0 [1, 256, 14, 14]
*/
void node__model_12_Concat(const float _model_11_Resize_output_0[1][128][14][14], const float _model_6_cv3_act_Mul_output_0[1][128][14][14], float _model_12_Concat_output_0[1][256][14][14]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_11_Resize_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_12_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_11_Resize_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
    // Copy tensor '_model_6_cv3_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_12_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_6_cv3_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
}

/*  Operator: Conv 
    Name in model: /model.13/cv1/conv/Conv
    Input: _model_12_Concat_output_0 [1, 256, 14, 14]
    Input: model_13_cv1_conv_weight [64, 256, 1, 1]
    Input: model_13_cv1_conv_bias [64]
    Output: _model_13_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_13_cv1_conv_Conv(const float _model_12_Concat_output_0[1][256][14][14], const float model_13_cv1_conv_weight[64][256][1][1], const float model_13_cv1_conv_bias[64], float _model_13_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_13_cv1_conv_bias[k];
                float ker_val = model_13_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_13_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_13_cv1_conv_Conv_output_0[n][k][h][w] += _model_12_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.13/cv2/conv/Conv
    Input: _model_12_Concat_output_0 [1, 256, 14, 14]
    Input: model_13_cv2_conv_weight [64, 256, 1, 1]
    Input: model_13_cv2_conv_bias [64]
    Output: _model_13_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_13_cv2_conv_Conv(const float _model_12_Concat_output_0[1][256][14][14], const float model_13_cv2_conv_weight[64][256][1][1], const float model_13_cv2_conv_bias[64], float _model_13_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_13_cv2_conv_bias[k];
                float ker_val = model_13_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_13_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_13_cv2_conv_Conv_output_0[n][k][h][w] += _model_12_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.13/cv1/act/Sigmoid
    Input: _model_13_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_13_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_13_cv1_act_Sigmoid(const float _model_13_cv1_conv_Conv_output_0[1][64][14][14], float _model_13_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_13_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_13_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.13/cv2/act/Sigmoid
    Input: _model_13_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_13_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_13_cv2_act_Sigmoid(const float _model_13_cv2_conv_Conv_output_0[1][64][14][14], float _model_13_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_13_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_13_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.13/cv1/act/Mul
    Input: _model_13_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_13_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_13_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_13_cv1_act_Mul(const float _model_13_cv1_conv_Conv_output_0[1][64][14][14], const float _model_13_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_13_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_13_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_13_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_13_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.13/cv2/act/Mul
    Input: _model_13_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_13_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_13_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_13_cv2_act_Mul(const float _model_13_cv2_conv_Conv_output_0[1][64][14][14], const float _model_13_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_13_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_13_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_13_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_13_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.13/m/m.0/cv1/conv/Conv
    Input: _model_13_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_13_m_0_cv1_conv_weight [64, 64, 1, 1]
    Input: model_13_m_0_cv1_conv_bias [64]
    Output: _model_13_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_13_m_m_0_cv1_conv_Conv(const float _model_13_cv1_act_Mul_output_0[1][64][14][14], const float model_13_m_0_cv1_conv_weight[64][64][1][1], const float model_13_m_0_cv1_conv_bias[64], float _model_13_m_m_0_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_13_m_0_cv1_conv_bias[k];
                float ker_val = model_13_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_13_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_13_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_13_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.13/m/m.0/cv1/act/Sigmoid
    Input: _model_13_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_13_m_m_0_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_13_m_m_0_cv1_act_Sigmoid(const float _model_13_m_m_0_cv1_conv_Conv_output_0[1][64][14][14], float _model_13_m_m_0_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_13_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_13_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.13/m/m.0/cv1/act/Mul
    Input: _model_13_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_13_m_m_0_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_13_m_m_0_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_13_m_m_0_cv1_act_Mul(const float _model_13_m_m_0_cv1_conv_Conv_output_0[1][64][14][14], const float _model_13_m_m_0_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_13_m_m_0_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_13_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_13_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_13_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.13/m/m.0/cv2/conv/Conv
    Input: _model_13_m_m_0_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_13_m_0_cv2_conv_weight [64, 64, 3, 3]
    Input: model_13_m_0_cv2_conv_bias [64]
    Output: _model_13_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_13_m_m_0_cv2_conv_Conv(const float _model_13_m_m_0_cv1_act_Mul_output_0[1][64][14][14], const float model_13_m_0_cv2_conv_weight[64][64][3][3], const float model_13_m_0_cv2_conv_bias[64], float _model_13_m_m_0_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 3, K_w = 3;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_13_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_13_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_13_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_13_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.13/m/m.0/cv2/act/Sigmoid
    Input: _model_13_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_13_m_m_0_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_13_m_m_0_cv2_act_Sigmoid(const float _model_13_m_m_0_cv2_conv_Conv_output_0[1][64][14][14], float _model_13_m_m_0_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_13_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_13_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.13/m/m.0/cv2/act/Mul
    Input: _model_13_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_13_m_m_0_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_13_m_m_0_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_13_m_m_0_cv2_act_Mul(const float _model_13_m_m_0_cv2_conv_Conv_output_0[1][64][14][14], const float _model_13_m_m_0_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_13_m_m_0_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_13_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_13_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_13_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.13/Concat
    Input: _model_13_m_m_0_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Input: _model_13_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Output: _model_13_Concat_output_0 [1, 128, 14, 14]
*/
void node__model_13_Concat(const float _model_13_m_m_0_cv2_act_Mul_output_0[1][64][14][14], const float _model_13_cv2_act_Mul_output_0[1][64][14][14], float _model_13_Concat_output_0[1][128][14][14]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_13_m_m_0_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_13_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_13_m_m_0_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
    // Copy tensor '_model_13_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_13_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_13_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
}

/*  Operator: Conv 
    Name in model: /model.13/cv3/conv/Conv
    Input: _model_13_Concat_output_0 [1, 128, 14, 14]
    Input: model_13_cv3_conv_weight [128, 128, 1, 1]
    Input: model_13_cv3_conv_bias [128]
    Output: _model_13_cv3_conv_Conv_output_0 [1, 128, 14, 14]
*/
void node__model_13_cv3_conv_Conv(const float _model_13_Concat_output_0[1][128][14][14], const float model_13_cv3_conv_weight[128][128][1][1], const float model_13_cv3_conv_bias[128], float _model_13_cv3_conv_Conv_output_0[1][128][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_13_cv3_conv_bias[k];
                float ker_val = model_13_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_13_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_13_cv3_conv_Conv_output_0[n][k][h][w] += _model_13_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.13/cv3/act/Sigmoid
    Input: _model_13_cv3_conv_Conv_output_0 [1, 128, 14, 14]
    Output: _model_13_cv3_act_Sigmoid_output_0 [1, 128, 14, 14]
*/
void node__model_13_cv3_act_Sigmoid(const float _model_13_cv3_conv_Conv_output_0[1][128][14][14], float _model_13_cv3_act_Sigmoid_output_0[1][128][14][14]) {
    float *X_ptr = (float *)_model_13_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_13_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.13/cv3/act/Mul
    Input: _model_13_cv3_conv_Conv_output_0 [1, 128, 14, 14]
    Input: _model_13_cv3_act_Sigmoid_output_0 [1, 128, 14, 14]
    Output: _model_13_cv3_act_Mul_output_0 [1, 128, 14, 14]
*/
void node__model_13_cv3_act_Mul(const float _model_13_cv3_conv_Conv_output_0[1][128][14][14], const float _model_13_cv3_act_Sigmoid_output_0[1][128][14][14], float _model_13_cv3_act_Mul_output_0[1][128][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_13_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_13_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_13_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.14/conv/Conv
    Input: _model_13_cv3_act_Mul_output_0 [1, 128, 14, 14]
    Input: model_14_conv_weight [64, 128, 1, 1]
    Input: model_14_conv_bias [64]
    Output: _model_14_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_14_conv_Conv(const float _model_13_cv3_act_Mul_output_0[1][128][14][14], const float model_14_conv_weight[64][128][1][1], const float model_14_conv_bias[64], float _model_14_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_14_conv_bias[k];
                float ker_val = model_14_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_14_conv_Conv_output_0[n][k][h][w] = b;
                        _model_14_conv_Conv_output_0[n][k][h][w] += _model_13_cv3_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.14/act/Sigmoid
    Input: _model_14_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_14_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_14_act_Sigmoid(const float _model_14_conv_Conv_output_0[1][64][14][14], float _model_14_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_14_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_14_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.14/act/Mul
    Input: _model_14_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_14_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_14_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_14_act_Mul(const float _model_14_conv_Conv_output_0[1][64][14][14], const float _model_14_act_Sigmoid_output_0[1][64][14][14], float _model_14_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_14_act_Mul_output_0[d0][d1][d2][d3] = _model_14_conv_Conv_output_0[d0][d1][d2][d3] * _model_14_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Resize 
    Name in model: /model.15/Resize
    Input: _model_14_act_Mul_output_0 [1, 64, 14, 14]
    Input:  None
    Input:  None
    Input: _model_15_Concat_1_output_0 [4]
    Output: _model_15_Resize_output_0 [1, 64, 28, 28]
*/
void node__model_15_Resize(const float _model_14_act_Mul_output_0[1][64][14][14], const int64_t _model_15_Concat_1_output_0[4], float _model_15_Resize_output_0[1][64][28][28]) {
    /*Resize: mode=b'nearest', coord_transform=b'asymmetric', cubic_a=-0.75, exclude_outside=0, extrapolation=0.0, nearest_mode=b'floor'*/
    const int N = 1;
    const int C = 64;
    const int H_in = 14;
    const int W_in = 14;
    const int H_out = _model_15_Concat_1_output_0[2]; // 28;
    const int W_out = _model_15_Concat_1_output_0[3]; // 28;

    float scale_h = (float)H_out / H_in;
    float scale_w = (float)W_out / W_in;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < H_out; ++y) {
                for (int x = 0; x < W_out; ++x) {
                    float y_in = ((float)y + 0.5f) / scale_h - 0.5f;
                    float x_in = ((float)x + 0.5f) / scale_w - 0.5f;
                    int y_index = (int)floorf(y_in);
                    int x_index = (int)floorf(x_in);
                    y_index = y_index < 0 ? 0 : (y_index >= H_in ? H_in - 1 : y_index);
                    x_index = x_index < 0 ? 0 : (x_index >= W_in ? W_in - 1 : x_index);
                    _model_15_Resize_output_0[n][c][y][x] = _model_14_act_Mul_output_0[n][c][y_index][x_index];
                }
            }
        }
    }
}

/*  Operator: Concat 
    Name in model: /model.16/Concat
    Input: _model_15_Resize_output_0 [1, 64, 28, 28]
    Input: _model_4_cv3_act_Mul_output_0 [1, 64, 28, 28]
    Output: _model_16_Concat_output_0 [1, 128, 28, 28]
*/
void node__model_16_Concat(const float _model_15_Resize_output_0[1][64][28][28], const float _model_4_cv3_act_Mul_output_0[1][64][28][28], float _model_16_Concat_output_0[1][128][28][28]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_15_Resize_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        _model_16_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_15_Resize_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
    // Copy tensor '_model_4_cv3_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        _model_16_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_4_cv3_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
}

/*  Operator: Conv 
    Name in model: /model.17/cv1/conv/Conv
    Input: _model_16_Concat_output_0 [1, 128, 28, 28]
    Input: model_17_cv1_conv_weight [32, 128, 1, 1]
    Input: model_17_cv1_conv_bias [32]
    Output: _model_17_cv1_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_17_cv1_conv_Conv(const float _model_16_Concat_output_0[1][128][28][28], const float model_17_cv1_conv_weight[32][128][1][1], const float model_17_cv1_conv_bias[32], float _model_17_cv1_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 28, W_in = 28;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_17_cv1_conv_bias[k];
                float ker_val = model_17_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_17_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_17_cv1_conv_Conv_output_0[n][k][h][w] += _model_16_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.17/cv2/conv/Conv
    Input: _model_16_Concat_output_0 [1, 128, 28, 28]
    Input: model_17_cv2_conv_weight [32, 128, 1, 1]
    Input: model_17_cv2_conv_bias [32]
    Output: _model_17_cv2_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_17_cv2_conv_Conv(const float _model_16_Concat_output_0[1][128][28][28], const float model_17_cv2_conv_weight[32][128][1][1], const float model_17_cv2_conv_bias[32], float _model_17_cv2_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 28, W_in = 28;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_17_cv2_conv_bias[k];
                float ker_val = model_17_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_17_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_17_cv2_conv_Conv_output_0[n][k][h][w] += _model_16_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.17/cv1/act/Sigmoid
    Input: _model_17_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_17_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_17_cv1_act_Sigmoid(const float _model_17_cv1_conv_Conv_output_0[1][32][28][28], float _model_17_cv1_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_17_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_17_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.17/cv2/act/Sigmoid
    Input: _model_17_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_17_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_17_cv2_act_Sigmoid(const float _model_17_cv2_conv_Conv_output_0[1][32][28][28], float _model_17_cv2_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_17_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_17_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.17/cv1/act/Mul
    Input: _model_17_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_17_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_17_cv1_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_17_cv1_act_Mul(const float _model_17_cv1_conv_Conv_output_0[1][32][28][28], const float _model_17_cv1_act_Sigmoid_output_0[1][32][28][28], float _model_17_cv1_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_17_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_17_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_17_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.17/cv2/act/Mul
    Input: _model_17_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_17_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_17_cv2_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_17_cv2_act_Mul(const float _model_17_cv2_conv_Conv_output_0[1][32][28][28], const float _model_17_cv2_act_Sigmoid_output_0[1][32][28][28], float _model_17_cv2_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_17_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_17_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_17_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.17/m/m.0/cv1/conv/Conv
    Input: _model_17_cv1_act_Mul_output_0 [1, 32, 28, 28]
    Input: model_17_m_0_cv1_conv_weight [32, 32, 1, 1]
    Input: model_17_m_0_cv1_conv_bias [32]
    Output: _model_17_m_m_0_cv1_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_17_m_m_0_cv1_conv_Conv(const float _model_17_cv1_act_Mul_output_0[1][32][28][28], const float model_17_m_0_cv1_conv_weight[32][32][1][1], const float model_17_m_0_cv1_conv_bias[32], float _model_17_m_m_0_cv1_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 28, W_in = 28;
    const int K = 32, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_17_m_0_cv1_conv_bias[k];
                float ker_val = model_17_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_17_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_17_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_17_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.17/m/m.0/cv1/act/Sigmoid
    Input: _model_17_m_m_0_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_17_m_m_0_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_17_m_m_0_cv1_act_Sigmoid(const float _model_17_m_m_0_cv1_conv_Conv_output_0[1][32][28][28], float _model_17_m_m_0_cv1_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_17_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_17_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.17/m/m.0/cv1/act/Mul
    Input: _model_17_m_m_0_cv1_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_17_m_m_0_cv1_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_17_m_m_0_cv1_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_17_m_m_0_cv1_act_Mul(const float _model_17_m_m_0_cv1_conv_Conv_output_0[1][32][28][28], const float _model_17_m_m_0_cv1_act_Sigmoid_output_0[1][32][28][28], float _model_17_m_m_0_cv1_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_17_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_17_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_17_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.17/m/m.0/cv2/conv/Conv
    Input: _model_17_m_m_0_cv1_act_Mul_output_0 [1, 32, 28, 28]
    Input: model_17_m_0_cv2_conv_weight [32, 32, 3, 3]
    Input: model_17_m_0_cv2_conv_bias [32]
    Output: _model_17_m_m_0_cv2_conv_Conv_output_0 [1, 32, 28, 28]
*/
void node__model_17_m_m_0_cv2_conv_Conv(const float _model_17_m_m_0_cv1_act_Mul_output_0[1][32][28][28], const float model_17_m_0_cv2_conv_weight[32][32][3][3], const float model_17_m_0_cv2_conv_bias[32], float _model_17_m_m_0_cv2_conv_Conv_output_0[1][32][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 32, H_in = 28, W_in = 28;
    const int K = 32, K_h = 3, K_w = 3;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_17_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_17_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_17_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_17_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.17/m/m.0/cv2/act/Sigmoid
    Input: _model_17_m_m_0_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Output: _model_17_m_m_0_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
*/
void node__model_17_m_m_0_cv2_act_Sigmoid(const float _model_17_m_m_0_cv2_conv_Conv_output_0[1][32][28][28], float _model_17_m_m_0_cv2_act_Sigmoid_output_0[1][32][28][28]) {
    float *X_ptr = (float *)_model_17_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_17_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.17/m/m.0/cv2/act/Mul
    Input: _model_17_m_m_0_cv2_conv_Conv_output_0 [1, 32, 28, 28]
    Input: _model_17_m_m_0_cv2_act_Sigmoid_output_0 [1, 32, 28, 28]
    Output: _model_17_m_m_0_cv2_act_Mul_output_0 [1, 32, 28, 28]
*/
void node__model_17_m_m_0_cv2_act_Mul(const float _model_17_m_m_0_cv2_conv_Conv_output_0[1][32][28][28], const float _model_17_m_m_0_cv2_act_Sigmoid_output_0[1][32][28][28], float _model_17_m_m_0_cv2_act_Mul_output_0[1][32][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 32; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_17_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_17_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_17_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.17/Concat
    Input: _model_17_m_m_0_cv2_act_Mul_output_0 [1, 32, 28, 28]
    Input: _model_17_cv2_act_Mul_output_0 [1, 32, 28, 28]
    Output: _model_17_Concat_output_0 [1, 64, 28, 28]
*/
void node__model_17_Concat(const float _model_17_m_m_0_cv2_act_Mul_output_0[1][32][28][28], const float _model_17_cv2_act_Mul_output_0[1][32][28][28], float _model_17_Concat_output_0[1][64][28][28]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_17_m_m_0_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 32; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        _model_17_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_17_m_m_0_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 32;
    // Copy tensor '_model_17_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 32; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        _model_17_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_17_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 32;
}

/*  Operator: Conv 
    Name in model: /model.17/cv3/conv/Conv
    Input: _model_17_Concat_output_0 [1, 64, 28, 28]
    Input: model_17_cv3_conv_weight [64, 64, 1, 1]
    Input: model_17_cv3_conv_bias [64]
    Output: _model_17_cv3_conv_Conv_output_0 [1, 64, 28, 28]
*/
void node__model_17_cv3_conv_Conv(const float _model_17_Concat_output_0[1][64][28][28], const float model_17_cv3_conv_weight[64][64][1][1], const float model_17_cv3_conv_bias[64], float _model_17_cv3_conv_Conv_output_0[1][64][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 28, W_in = 28;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_17_cv3_conv_bias[k];
                float ker_val = model_17_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_17_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_17_cv3_conv_Conv_output_0[n][k][h][w] += _model_17_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.17/cv3/act/Sigmoid
    Input: _model_17_cv3_conv_Conv_output_0 [1, 64, 28, 28]
    Output: _model_17_cv3_act_Sigmoid_output_0 [1, 64, 28, 28]
*/
void node__model_17_cv3_act_Sigmoid(const float _model_17_cv3_conv_Conv_output_0[1][64][28][28], float _model_17_cv3_act_Sigmoid_output_0[1][64][28][28]) {
    float *X_ptr = (float *)_model_17_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_17_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 50176; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.17/cv3/act/Mul
    Input: _model_17_cv3_conv_Conv_output_0 [1, 64, 28, 28]
    Input: _model_17_cv3_act_Sigmoid_output_0 [1, 64, 28, 28]
    Output: _model_17_cv3_act_Mul_output_0 [1, 64, 28, 28]
*/
void node__model_17_cv3_act_Mul(const float _model_17_cv3_conv_Conv_output_0[1][64][28][28], const float _model_17_cv3_act_Sigmoid_output_0[1][64][28][28], float _model_17_cv3_act_Mul_output_0[1][64][28][28]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
        _model_17_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_17_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_17_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.18/conv/Conv
    Input: _model_17_cv3_act_Mul_output_0 [1, 64, 28, 28]
    Input: model_18_conv_weight [64, 64, 3, 3]
    Input: model_18_conv_bias [64]
    Output: _model_18_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_18_conv_Conv(const float _model_17_cv3_act_Mul_output_0[1][64][28][28], const float model_18_conv_weight[64][64][3][3], const float model_18_conv_bias[64], float _model_18_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 28, W_in = 28;
    const int K = 64, K_h = 3, K_w = 3;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 2, stride_w = 2;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_18_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_17_cv3_act_Mul_output_0[n][c][h_in][w_in] * model_18_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_18_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.24/m.0/Conv
    Input: _model_17_cv3_act_Mul_output_0 [1, 64, 28, 28]
    Input: model_24_m_0_weight [255, 64, 1, 1]
    Input: model_24_m_0_bias [255]
    Output: _model_24_m_0_Conv_output_0 [1, 255, 28, 28]
*/
void node__model_24_m_0_Conv(const float _model_17_cv3_act_Mul_output_0[1][64][28][28], const float model_24_m_0_weight[255][64][1][1], const float model_24_m_0_bias[255], float _model_24_m_0_Conv_output_0[1][255][28][28]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 28, W_in = 28;
    const int K = 255, K_h = 1, K_w = 1;
    const int H_out = 28, W_out = 28;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_24_m_0_bias[k];
                float ker_val = model_24_m_0_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_24_m_0_Conv_output_0[n][k][h][w] = b;
                        _model_24_m_0_Conv_output_0[n][k][h][w] += _model_17_cv3_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.18/act/Sigmoid
    Input: _model_18_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_18_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_18_act_Sigmoid(const float _model_18_conv_Conv_output_0[1][64][14][14], float _model_18_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_18_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_18_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Reshape 
    Name in model: /model.24/Reshape
    Input: _model_24_m_0_Conv_output_0 [1, 255, 28, 28]
    Input: _model_24_Constant_output_0 [5]
    Output: _model_24_Reshape_output_0 [1, 3, 85, 28, 28]
*/
void node__model_24_Reshape(const float _model_24_m_0_Conv_output_0[1][255][28][28], const int64_t _model_24_Constant_output_0[5], float _model_24_Reshape_output_0[1][3][85][28][28]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_model_24_m_0_Conv_output_0;
    float *dst = (float*)_model_24_Reshape_output_0;
    memcpy(dst, src, 199920 * sizeof(float));
}

/*  Operator: Mul 
    Name in model: /model.18/act/Mul
    Input: _model_18_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_18_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_18_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_18_act_Mul(const float _model_18_conv_Conv_output_0[1][64][14][14], const float _model_18_act_Sigmoid_output_0[1][64][14][14], float _model_18_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_18_act_Mul_output_0[d0][d1][d2][d3] = _model_18_conv_Conv_output_0[d0][d1][d2][d3] * _model_18_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Transpose 
    Name in model: /model.24/Transpose
    Input: _model_24_Reshape_output_0 [1, 3, 85, 28, 28]
    Output: _model_24_Transpose_output_0 [1, 3, 28, 28, 85]
*/
void node__model_24_Transpose(const float _model_24_Reshape_output_0[1][3][85][28][28], float _model_24_Transpose_output_0[1][3][28][28][85]) {
    /*Transpose with perm = [0, 1, 3, 4, 2]*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 28; i2++) {
                for (int i3 = 0; i3 < 28; i3++) {
                    for (int i4 = 0; i4 < 85; i4++) {
                        _model_24_Transpose_output_0[i0][i1][i2][i3][i4] = _model_24_Reshape_output_0[i0][i1][i4][i2][i3];
                    }
                }
            }
        }
    }
}

/*  Operator: Concat 
    Name in model: /model.19/Concat
    Input: _model_18_act_Mul_output_0 [1, 64, 14, 14]
    Input: _model_14_act_Mul_output_0 [1, 64, 14, 14]
    Output: _model_19_Concat_output_0 [1, 128, 14, 14]
*/
void node__model_19_Concat(const float _model_18_act_Mul_output_0[1][64][14][14], const float _model_14_act_Mul_output_0[1][64][14][14], float _model_19_Concat_output_0[1][128][14][14]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_18_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_19_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_18_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
    // Copy tensor '_model_14_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_19_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_14_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
}

/*  Operator: Sigmoid 
    Name in model: /model.24/Sigmoid
    Input: _model_24_Transpose_output_0 [1, 3, 28, 28, 85]
    Output: _model_24_Sigmoid_output_0 [1, 3, 28, 28, 85]
*/
void node__model_24_Sigmoid(const float _model_24_Transpose_output_0[1][3][28][28][85], float _model_24_Sigmoid_output_0[1][3][28][28][85]) {
    float *X_ptr = (float *)_model_24_Transpose_output_0;
    float *Y_ptr = (float *)_model_24_Sigmoid_output_0;
    for (int i = 0; i < 199920; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Conv 
    Name in model: /model.20/cv1/conv/Conv
    Input: _model_19_Concat_output_0 [1, 128, 14, 14]
    Input: model_20_cv1_conv_weight [64, 128, 1, 1]
    Input: model_20_cv1_conv_bias [64]
    Output: _model_20_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_20_cv1_conv_Conv(const float _model_19_Concat_output_0[1][128][14][14], const float model_20_cv1_conv_weight[64][128][1][1], const float model_20_cv1_conv_bias[64], float _model_20_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_20_cv1_conv_bias[k];
                float ker_val = model_20_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_20_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_20_cv1_conv_Conv_output_0[n][k][h][w] += _model_19_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.20/cv2/conv/Conv
    Input: _model_19_Concat_output_0 [1, 128, 14, 14]
    Input: model_20_cv2_conv_weight [64, 128, 1, 1]
    Input: model_20_cv2_conv_bias [64]
    Output: _model_20_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_20_cv2_conv_Conv(const float _model_19_Concat_output_0[1][128][14][14], const float model_20_cv2_conv_weight[64][128][1][1], const float model_20_cv2_conv_bias[64], float _model_20_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_20_cv2_conv_bias[k];
                float ker_val = model_20_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_20_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_20_cv2_conv_Conv_output_0[n][k][h][w] += _model_19_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Split 
    Name in model: /model.24/Split
    Input: _model_24_Sigmoid_output_0 [1, 3, 28, 28, 85]
    Input: onnx__Split_377 [3]
    Output: _model_24_Split_output_0 [1, 3, 28, 28, 2]    Output: _model_24_Split_output_1 [1, 3, 28, 28, 2]    Output: _model_24_Split_output_2 [1, 3, 28, 28, 81]
*/
void node__model_24_Split(const float _model_24_Sigmoid_output_0[1][3][28][28][85], const int64_t onnx__Split_377[3], float _model_24_Split_output_0[1][3][28][28][2], float _model_24_Split_output_1[1][3][28][28][2], float _model_24_Split_output_2[1][3][28][28][81]) {
    // Split along axis=4
    const int64_t* split = (const int64_t*)onnx__Split_377;
    // Processing output 0: _model_24_Split_output_0
    const int64_t start_0 = 0;
    const int64_t split_size_0 = split[0];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 28; i2++) {
                for (int i3 = 0; i3 < 28; i3++) {
                    for (int i4 = 0; i4 < split_size_0; i4++) {
                        _model_24_Split_output_0[i0][i1][i2][i3][i4] = _model_24_Sigmoid_output_0[i0][i1][i2][i3][start_0 + i4];
                    }
                }
            }
        }
    }
    // Processing output 1: _model_24_Split_output_1
    const int64_t start_1 = split[0];
    const int64_t split_size_1 = split[1];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 28; i2++) {
                for (int i3 = 0; i3 < 28; i3++) {
                    for (int i4 = 0; i4 < split_size_1; i4++) {
                        _model_24_Split_output_1[i0][i1][i2][i3][i4] = _model_24_Sigmoid_output_0[i0][i1][i2][i3][start_1 + i4];
                    }
                }
            }
        }
    }
    // Processing output 2: _model_24_Split_output_2
    const int64_t start_2 = split[0] + split[1];
    const int64_t split_size_2 = split[2];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 28; i2++) {
                for (int i3 = 0; i3 < 28; i3++) {
                    for (int i4 = 0; i4 < split_size_2; i4++) {
                        _model_24_Split_output_2[i0][i1][i2][i3][i4] = _model_24_Sigmoid_output_0[i0][i1][i2][i3][start_2 + i4];
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.20/cv1/act/Sigmoid
    Input: _model_20_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_20_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_20_cv1_act_Sigmoid(const float _model_20_cv1_conv_Conv_output_0[1][64][14][14], float _model_20_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_20_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_20_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.20/cv2/act/Sigmoid
    Input: _model_20_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_20_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_20_cv2_act_Sigmoid(const float _model_20_cv2_conv_Conv_output_0[1][64][14][14], float _model_20_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_20_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_20_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul
    Input: _model_24_Split_output_0 [1, 3, 28, 28, 2]
    Input: _model_24_Constant_1_output_0 [1]
    Output: _model_24_Mul_output_0 [1, 3, 28, 28, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul(const float _model_24_Split_output_0[1][3][28][28][2], const float _model_24_Constant_1_output_0[1], float _model_24_Mul_output_0[1][3][28][28][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_output_0[d0][d1][d2][d3][d4] = _model_24_Split_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_1_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_2
    Input: _model_24_Split_output_1 [1, 3, 28, 28, 2]
    Input: _model_24_Constant_1_output_0 [1]
    Output: _model_24_Mul_2_output_0 [1, 3, 28, 28, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_2(const float _model_24_Split_output_1[1][3][28][28][2], const float _model_24_Constant_1_output_0[1], float _model_24_Mul_2_output_0[1][3][28][28][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_2_output_0[d0][d1][d2][d3][d4] = _model_24_Split_output_1[d0][d1][d2][d3][d4] * _model_24_Constant_1_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.20/cv1/act/Mul
    Input: _model_20_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_20_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_20_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_20_cv1_act_Mul(const float _model_20_cv1_conv_Conv_output_0[1][64][14][14], const float _model_20_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_20_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_20_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_20_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_20_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.20/cv2/act/Mul
    Input: _model_20_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_20_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_20_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_20_cv2_act_Mul(const float _model_20_cv2_conv_Conv_output_0[1][64][14][14], const float _model_20_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_20_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_20_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_20_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_20_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.24/Add
    Input: _model_24_Mul_output_0 [1, 3, 28, 28, 2]
    Input: _model_24_Constant_2_output_0 [1, 3, 28, 28, 2]
    Output: _model_24_Add_output_0 [1, 3, 28, 28, 2]
*/
void node__model_24_Add(const float _model_24_Mul_output_0[1][3][28][28][2], const float _model_24_Constant_2_output_0[1][3][28][28][2], float _model_24_Add_output_0[1][3][28][28][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Add_output_0[d0][d1][d2][d3][d4] = _model_24_Mul_output_0[d0][d1][d2][d3][d4] + _model_24_Constant_2_output_0[d0][d1][d2][d3][d4];
    }
    }
    }
    }
    }
}

/*  Operator: Pow 
    Name in model: /model.24/Pow
    Input: _model_24_Mul_2_output_0 [1, 3, 28, 28, 2]
    Input: _model_24_Constant_5_output_0 [1]
    Output: _model_24_Pow_output_0 [1, 3, 28, 28, 2]
*/
void node__model_24_Pow(const float _model_24_Mul_2_output_0[1][3][28][28][2], const float _model_24_Constant_5_output_0[1], float _model_24_Pow_output_0[1][3][28][28][2]) {
    /*pow*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 28; i2++) {
                for (int i3 = 0; i3 < 28; i3++) {
                    for (int i4 = 0; i4 < 2; i4++) {
                        _model_24_Pow_output_0[i0][i1][i2][i3][i4] = pow(_model_24_Mul_2_output_0[i0][i1][i2][i3][i4], _model_24_Constant_5_output_0[0]);
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.20/m/m.0/cv1/conv/Conv
    Input: _model_20_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_20_m_0_cv1_conv_weight [64, 64, 1, 1]
    Input: model_20_m_0_cv1_conv_bias [64]
    Output: _model_20_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_20_m_m_0_cv1_conv_Conv(const float _model_20_cv1_act_Mul_output_0[1][64][14][14], const float model_20_m_0_cv1_conv_weight[64][64][1][1], const float model_20_m_0_cv1_conv_bias[64], float _model_20_m_m_0_cv1_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_20_m_0_cv1_conv_bias[k];
                float ker_val = model_20_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_20_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_20_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_20_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_1
    Input: _model_24_Add_output_0 [1, 3, 28, 28, 2]
    Input: _model_24_Constant_3_output_0 [1]
    Output: _model_24_Mul_1_output_0 [1, 3, 28, 28, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_1(const float _model_24_Add_output_0[1][3][28][28][2], const float _model_24_Constant_3_output_0[1], float _model_24_Mul_1_output_0[1][3][28][28][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_1_output_0[d0][d1][d2][d3][d4] = _model_24_Add_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_3_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_3
    Input: _model_24_Pow_output_0 [1, 3, 28, 28, 2]
    Input: _model_24_Constant_6_output_0 [1, 3, 28, 28, 2]
    Output: _model_24_Mul_3_output_0 [1, 3, 28, 28, 2]
*/
void node__model_24_Mul_3(const float _model_24_Pow_output_0[1][3][28][28][2], const float _model_24_Constant_6_output_0[1][3][28][28][2], float _model_24_Mul_3_output_0[1][3][28][28][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 28; d2++) {
    for (int d3 = 0; d3 < 28; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_3_output_0[d0][d1][d2][d3][d4] = _model_24_Pow_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_6_output_0[d0][d1][d2][d3][d4];
    }
    }
    }
    }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.20/m/m.0/cv1/act/Sigmoid
    Input: _model_20_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_20_m_m_0_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_20_m_m_0_cv1_act_Sigmoid(const float _model_20_m_m_0_cv1_conv_Conv_output_0[1][64][14][14], float _model_20_m_m_0_cv1_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_20_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_20_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Concat 
    Name in model: /model.24/Concat
    Input: _model_24_Mul_1_output_0 [1, 3, 28, 28, 2]
    Input: _model_24_Mul_3_output_0 [1, 3, 28, 28, 2]
    Input: _model_24_Split_output_2 [1, 3, 28, 28, 81]
    Output: _model_24_Concat_output_0 [1, 3, 28, 28, 85]
*/
void node__model_24_Concat(const float _model_24_Mul_1_output_0[1][3][28][28][2], const float _model_24_Mul_3_output_0[1][3][28][28][2], const float _model_24_Split_output_2[1][3][28][28][81], float _model_24_Concat_output_0[1][3][28][28][85]) {
    /*Concat along axis=4*/
    int axis_offset = 0;
    // Copy tensor '_model_24_Mul_1_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        for (int i4 = 0; i4 < 2; i4++) {
                            _model_24_Concat_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Mul_1_output_0[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 2;
    // Copy tensor '_model_24_Mul_3_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        for (int i4 = 0; i4 < 2; i4++) {
                            _model_24_Concat_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Mul_3_output_0[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 2;
    // Copy tensor '_model_24_Split_output_2' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 28; i2++) {
                    for (int i3 = 0; i3 < 28; i3++) {
                        for (int i4 = 0; i4 < 81; i4++) {
                            _model_24_Concat_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Split_output_2[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 81;
}

/*  Operator: Mul 
    Name in model: /model.20/m/m.0/cv1/act/Mul
    Input: _model_20_m_m_0_cv1_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_20_m_m_0_cv1_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_20_m_m_0_cv1_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_20_m_m_0_cv1_act_Mul(const float _model_20_m_m_0_cv1_conv_Conv_output_0[1][64][14][14], const float _model_20_m_m_0_cv1_act_Sigmoid_output_0[1][64][14][14], float _model_20_m_m_0_cv1_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_20_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_20_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_20_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Reshape 
    Name in model: /model.24/Reshape_1
    Input: _model_24_Concat_output_0 [1, 3, 28, 28, 85]
    Input: _model_24_Constant_7_output_0 [3]
    Output: _model_24_Reshape_1_output_0 [1, 2352, 85]
*/
void node__model_24_Reshape_1(const float _model_24_Concat_output_0[1][3][28][28][85], const int64_t _model_24_Constant_7_output_0[3], float _model_24_Reshape_1_output_0[1][2352][85]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_model_24_Concat_output_0;
    float *dst = (float*)_model_24_Reshape_1_output_0;
    memcpy(dst, src, 199920 * sizeof(float));
}

/*  Operator: Conv 
    Name in model: /model.20/m/m.0/cv2/conv/Conv
    Input: _model_20_m_m_0_cv1_act_Mul_output_0 [1, 64, 14, 14]
    Input: model_20_m_0_cv2_conv_weight [64, 64, 3, 3]
    Input: model_20_m_0_cv2_conv_bias [64]
    Output: _model_20_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
*/
void node__model_20_m_m_0_cv2_conv_Conv(const float _model_20_m_m_0_cv1_act_Mul_output_0[1][64][14][14], const float model_20_m_0_cv2_conv_weight[64][64][3][3], const float model_20_m_0_cv2_conv_bias[64], float _model_20_m_m_0_cv2_conv_Conv_output_0[1][64][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 64, H_in = 14, W_in = 14;
    const int K = 64, K_h = 3, K_w = 3;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_20_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_20_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_20_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_20_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.20/m/m.0/cv2/act/Sigmoid
    Input: _model_20_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Output: _model_20_m_m_0_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
*/
void node__model_20_m_m_0_cv2_act_Sigmoid(const float _model_20_m_m_0_cv2_conv_Conv_output_0[1][64][14][14], float _model_20_m_m_0_cv2_act_Sigmoid_output_0[1][64][14][14]) {
    float *X_ptr = (float *)_model_20_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_20_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.20/m/m.0/cv2/act/Mul
    Input: _model_20_m_m_0_cv2_conv_Conv_output_0 [1, 64, 14, 14]
    Input: _model_20_m_m_0_cv2_act_Sigmoid_output_0 [1, 64, 14, 14]
    Output: _model_20_m_m_0_cv2_act_Mul_output_0 [1, 64, 14, 14]
*/
void node__model_20_m_m_0_cv2_act_Mul(const float _model_20_m_m_0_cv2_conv_Conv_output_0[1][64][14][14], const float _model_20_m_m_0_cv2_act_Sigmoid_output_0[1][64][14][14], float _model_20_m_m_0_cv2_act_Mul_output_0[1][64][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_20_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_20_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_20_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.20/Concat
    Input: _model_20_m_m_0_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Input: _model_20_cv2_act_Mul_output_0 [1, 64, 14, 14]
    Output: _model_20_Concat_output_0 [1, 128, 14, 14]
*/
void node__model_20_Concat(const float _model_20_m_m_0_cv2_act_Mul_output_0[1][64][14][14], const float _model_20_cv2_act_Mul_output_0[1][64][14][14], float _model_20_Concat_output_0[1][128][14][14]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_20_m_m_0_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_20_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_20_m_m_0_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
    // Copy tensor '_model_20_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 64; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        _model_20_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_20_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 64;
}

/*  Operator: Conv 
    Name in model: /model.20/cv3/conv/Conv
    Input: _model_20_Concat_output_0 [1, 128, 14, 14]
    Input: model_20_cv3_conv_weight [128, 128, 1, 1]
    Input: model_20_cv3_conv_bias [128]
    Output: _model_20_cv3_conv_Conv_output_0 [1, 128, 14, 14]
*/
void node__model_20_cv3_conv_Conv(const float _model_20_Concat_output_0[1][128][14][14], const float model_20_cv3_conv_weight[128][128][1][1], const float model_20_cv3_conv_bias[128], float _model_20_cv3_conv_Conv_output_0[1][128][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_20_cv3_conv_bias[k];
                float ker_val = model_20_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_20_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_20_cv3_conv_Conv_output_0[n][k][h][w] += _model_20_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.20/cv3/act/Sigmoid
    Input: _model_20_cv3_conv_Conv_output_0 [1, 128, 14, 14]
    Output: _model_20_cv3_act_Sigmoid_output_0 [1, 128, 14, 14]
*/
void node__model_20_cv3_act_Sigmoid(const float _model_20_cv3_conv_Conv_output_0[1][128][14][14], float _model_20_cv3_act_Sigmoid_output_0[1][128][14][14]) {
    float *X_ptr = (float *)_model_20_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_20_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 25088; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.20/cv3/act/Mul
    Input: _model_20_cv3_conv_Conv_output_0 [1, 128, 14, 14]
    Input: _model_20_cv3_act_Sigmoid_output_0 [1, 128, 14, 14]
    Output: _model_20_cv3_act_Mul_output_0 [1, 128, 14, 14]
*/
void node__model_20_cv3_act_Mul(const float _model_20_cv3_conv_Conv_output_0[1][128][14][14], const float _model_20_cv3_act_Sigmoid_output_0[1][128][14][14], float _model_20_cv3_act_Mul_output_0[1][128][14][14]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
        _model_20_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_20_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_20_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.21/conv/Conv
    Input: _model_20_cv3_act_Mul_output_0 [1, 128, 14, 14]
    Input: model_21_conv_weight [128, 128, 3, 3]
    Input: model_21_conv_bias [128]
    Output: _model_21_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_21_conv_Conv(const float _model_20_cv3_act_Mul_output_0[1][128][14][14], const float model_21_conv_weight[128][128][3][3], const float model_21_conv_bias[128], float _model_21_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 128, K_h = 3, K_w = 3;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 2, stride_w = 2;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_21_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_20_cv3_act_Mul_output_0[n][c][h_in][w_in] * model_21_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_21_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.24/m.1/Conv
    Input: _model_20_cv3_act_Mul_output_0 [1, 128, 14, 14]
    Input: model_24_m_1_weight [255, 128, 1, 1]
    Input: model_24_m_1_bias [255]
    Output: _model_24_m_1_Conv_output_0 [1, 255, 14, 14]
*/
void node__model_24_m_1_Conv(const float _model_20_cv3_act_Mul_output_0[1][128][14][14], const float model_24_m_1_weight[255][128][1][1], const float model_24_m_1_bias[255], float _model_24_m_1_Conv_output_0[1][255][14][14]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 14, W_in = 14;
    const int K = 255, K_h = 1, K_w = 1;
    const int H_out = 14, W_out = 14;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_24_m_1_bias[k];
                float ker_val = model_24_m_1_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_24_m_1_Conv_output_0[n][k][h][w] = b;
                        _model_24_m_1_Conv_output_0[n][k][h][w] += _model_20_cv3_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.21/act/Sigmoid
    Input: _model_21_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_21_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_21_act_Sigmoid(const float _model_21_conv_Conv_output_0[1][128][7][7], float _model_21_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_21_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_21_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Reshape 
    Name in model: /model.24/Reshape_2
    Input: _model_24_m_1_Conv_output_0 [1, 255, 14, 14]
    Input: _model_24_Constant_8_output_0 [5]
    Output: _model_24_Reshape_2_output_0 [1, 3, 85, 14, 14]
*/
void node__model_24_Reshape_2(const float _model_24_m_1_Conv_output_0[1][255][14][14], const int64_t _model_24_Constant_8_output_0[5], float _model_24_Reshape_2_output_0[1][3][85][14][14]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_model_24_m_1_Conv_output_0;
    float *dst = (float*)_model_24_Reshape_2_output_0;
    memcpy(dst, src, 49980 * sizeof(float));
}

/*  Operator: Mul 
    Name in model: /model.21/act/Mul
    Input: _model_21_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_21_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_21_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_21_act_Mul(const float _model_21_conv_Conv_output_0[1][128][7][7], const float _model_21_act_Sigmoid_output_0[1][128][7][7], float _model_21_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_21_act_Mul_output_0[d0][d1][d2][d3] = _model_21_conv_Conv_output_0[d0][d1][d2][d3] * _model_21_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Transpose 
    Name in model: /model.24/Transpose_1
    Input: _model_24_Reshape_2_output_0 [1, 3, 85, 14, 14]
    Output: _model_24_Transpose_1_output_0 [1, 3, 14, 14, 85]
*/
void node__model_24_Transpose_1(const float _model_24_Reshape_2_output_0[1][3][85][14][14], float _model_24_Transpose_1_output_0[1][3][14][14][85]) {
    /*Transpose with perm = [0, 1, 3, 4, 2]*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 14; i2++) {
                for (int i3 = 0; i3 < 14; i3++) {
                    for (int i4 = 0; i4 < 85; i4++) {
                        _model_24_Transpose_1_output_0[i0][i1][i2][i3][i4] = _model_24_Reshape_2_output_0[i0][i1][i4][i2][i3];
                    }
                }
            }
        }
    }
}

/*  Operator: Concat 
    Name in model: /model.22/Concat
    Input: _model_21_act_Mul_output_0 [1, 128, 7, 7]
    Input: _model_10_act_Mul_output_0 [1, 128, 7, 7]
    Output: _model_22_Concat_output_0 [1, 256, 7, 7]
*/
void node__model_22_Concat(const float _model_21_act_Mul_output_0[1][128][7][7], const float _model_10_act_Mul_output_0[1][128][7][7], float _model_22_Concat_output_0[1][256][7][7]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_21_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_22_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_21_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
    // Copy tensor '_model_10_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_22_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_10_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
}

/*  Operator: Sigmoid 
    Name in model: /model.24/Sigmoid_1
    Input: _model_24_Transpose_1_output_0 [1, 3, 14, 14, 85]
    Output: _model_24_Sigmoid_1_output_0 [1, 3, 14, 14, 85]
*/
void node__model_24_Sigmoid_1(const float _model_24_Transpose_1_output_0[1][3][14][14][85], float _model_24_Sigmoid_1_output_0[1][3][14][14][85]) {
    float *X_ptr = (float *)_model_24_Transpose_1_output_0;
    float *Y_ptr = (float *)_model_24_Sigmoid_1_output_0;
    for (int i = 0; i < 49980; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Conv 
    Name in model: /model.23/cv1/conv/Conv
    Input: _model_22_Concat_output_0 [1, 256, 7, 7]
    Input: model_23_cv1_conv_weight [128, 256, 1, 1]
    Input: model_23_cv1_conv_bias [128]
    Output: _model_23_cv1_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_23_cv1_conv_Conv(const float _model_22_Concat_output_0[1][256][7][7], const float model_23_cv1_conv_weight[128][256][1][1], const float model_23_cv1_conv_bias[128], float _model_23_cv1_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_23_cv1_conv_bias[k];
                float ker_val = model_23_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_23_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_23_cv1_conv_Conv_output_0[n][k][h][w] += _model_22_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.23/cv2/conv/Conv
    Input: _model_22_Concat_output_0 [1, 256, 7, 7]
    Input: model_23_cv2_conv_weight [128, 256, 1, 1]
    Input: model_23_cv2_conv_bias [128]
    Output: _model_23_cv2_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_23_cv2_conv_Conv(const float _model_22_Concat_output_0[1][256][7][7], const float model_23_cv2_conv_weight[128][256][1][1], const float model_23_cv2_conv_bias[128], float _model_23_cv2_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_23_cv2_conv_bias[k];
                float ker_val = model_23_cv2_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_23_cv2_conv_Conv_output_0[n][k][h][w] = b;
                        _model_23_cv2_conv_Conv_output_0[n][k][h][w] += _model_22_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Split 
    Name in model: /model.24/Split_1
    Input: _model_24_Sigmoid_1_output_0 [1, 3, 14, 14, 85]
    Input: onnx__Split_377 [3]
    Output: _model_24_Split_1_output_0 [1, 3, 14, 14, 2]    Output: _model_24_Split_1_output_1 [1, 3, 14, 14, 2]    Output: _model_24_Split_1_output_2 [1, 3, 14, 14, 81]
*/
void node__model_24_Split_1(const float _model_24_Sigmoid_1_output_0[1][3][14][14][85], const int64_t onnx__Split_377[3], float _model_24_Split_1_output_0[1][3][14][14][2], float _model_24_Split_1_output_1[1][3][14][14][2], float _model_24_Split_1_output_2[1][3][14][14][81]) {
    // Split along axis=4
    const int64_t* split = (const int64_t*)onnx__Split_377;
    // Processing output 0: _model_24_Split_1_output_0
    const int64_t start_0 = 0;
    const int64_t split_size_0 = split[0];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 14; i2++) {
                for (int i3 = 0; i3 < 14; i3++) {
                    for (int i4 = 0; i4 < split_size_0; i4++) {
                        _model_24_Split_1_output_0[i0][i1][i2][i3][i4] = _model_24_Sigmoid_1_output_0[i0][i1][i2][i3][start_0 + i4];
                    }
                }
            }
        }
    }
    // Processing output 1: _model_24_Split_1_output_1
    const int64_t start_1 = split[0];
    const int64_t split_size_1 = split[1];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 14; i2++) {
                for (int i3 = 0; i3 < 14; i3++) {
                    for (int i4 = 0; i4 < split_size_1; i4++) {
                        _model_24_Split_1_output_1[i0][i1][i2][i3][i4] = _model_24_Sigmoid_1_output_0[i0][i1][i2][i3][start_1 + i4];
                    }
                }
            }
        }
    }
    // Processing output 2: _model_24_Split_1_output_2
    const int64_t start_2 = split[0] + split[1];
    const int64_t split_size_2 = split[2];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 14; i2++) {
                for (int i3 = 0; i3 < 14; i3++) {
                    for (int i4 = 0; i4 < split_size_2; i4++) {
                        _model_24_Split_1_output_2[i0][i1][i2][i3][i4] = _model_24_Sigmoid_1_output_0[i0][i1][i2][i3][start_2 + i4];
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.23/cv1/act/Sigmoid
    Input: _model_23_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_23_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_23_cv1_act_Sigmoid(const float _model_23_cv1_conv_Conv_output_0[1][128][7][7], float _model_23_cv1_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_23_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_23_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.23/cv2/act/Sigmoid
    Input: _model_23_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_23_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_23_cv2_act_Sigmoid(const float _model_23_cv2_conv_Conv_output_0[1][128][7][7], float _model_23_cv2_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_23_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_23_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_4
    Input: _model_24_Split_1_output_0 [1, 3, 14, 14, 2]
    Input: _model_24_Constant_1_output_0 [1]
    Output: _model_24_Mul_4_output_0 [1, 3, 14, 14, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_4(const float _model_24_Split_1_output_0[1][3][14][14][2], const float _model_24_Constant_1_output_0[1], float _model_24_Mul_4_output_0[1][3][14][14][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_4_output_0[d0][d1][d2][d3][d4] = _model_24_Split_1_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_1_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_6
    Input: _model_24_Split_1_output_1 [1, 3, 14, 14, 2]
    Input: _model_24_Constant_1_output_0 [1]
    Output: _model_24_Mul_6_output_0 [1, 3, 14, 14, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_6(const float _model_24_Split_1_output_1[1][3][14][14][2], const float _model_24_Constant_1_output_0[1], float _model_24_Mul_6_output_0[1][3][14][14][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_6_output_0[d0][d1][d2][d3][d4] = _model_24_Split_1_output_1[d0][d1][d2][d3][d4] * _model_24_Constant_1_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.23/cv1/act/Mul
    Input: _model_23_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_23_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_23_cv1_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_23_cv1_act_Mul(const float _model_23_cv1_conv_Conv_output_0[1][128][7][7], const float _model_23_cv1_act_Sigmoid_output_0[1][128][7][7], float _model_23_cv1_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_23_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_23_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_23_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.23/cv2/act/Mul
    Input: _model_23_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_23_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_23_cv2_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_23_cv2_act_Mul(const float _model_23_cv2_conv_Conv_output_0[1][128][7][7], const float _model_23_cv2_act_Sigmoid_output_0[1][128][7][7], float _model_23_cv2_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_23_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_23_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_23_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.24/Add_1
    Input: _model_24_Mul_4_output_0 [1, 3, 14, 14, 2]
    Input: _model_24_Constant_10_output_0 [1, 3, 14, 14, 2]
    Output: _model_24_Add_1_output_0 [1, 3, 14, 14, 2]
*/
void node__model_24_Add_1(const float _model_24_Mul_4_output_0[1][3][14][14][2], const float _model_24_Constant_10_output_0[1][3][14][14][2], float _model_24_Add_1_output_0[1][3][14][14][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Add_1_output_0[d0][d1][d2][d3][d4] = _model_24_Mul_4_output_0[d0][d1][d2][d3][d4] + _model_24_Constant_10_output_0[d0][d1][d2][d3][d4];
    }
    }
    }
    }
    }
}

/*  Operator: Pow 
    Name in model: /model.24/Pow_1
    Input: _model_24_Mul_6_output_0 [1, 3, 14, 14, 2]
    Input: _model_24_Constant_5_output_0 [1]
    Output: _model_24_Pow_1_output_0 [1, 3, 14, 14, 2]
*/
void node__model_24_Pow_1(const float _model_24_Mul_6_output_0[1][3][14][14][2], const float _model_24_Constant_5_output_0[1], float _model_24_Pow_1_output_0[1][3][14][14][2]) {
    /*pow*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 14; i2++) {
                for (int i3 = 0; i3 < 14; i3++) {
                    for (int i4 = 0; i4 < 2; i4++) {
                        _model_24_Pow_1_output_0[i0][i1][i2][i3][i4] = pow(_model_24_Mul_6_output_0[i0][i1][i2][i3][i4], _model_24_Constant_5_output_0[0]);
                    }
                }
            }
        }
    }
}

/*  Operator: Conv 
    Name in model: /model.23/m/m.0/cv1/conv/Conv
    Input: _model_23_cv1_act_Mul_output_0 [1, 128, 7, 7]
    Input: model_23_m_0_cv1_conv_weight [128, 128, 1, 1]
    Input: model_23_m_0_cv1_conv_bias [128]
    Output: _model_23_m_m_0_cv1_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_23_m_m_0_cv1_conv_Conv(const float _model_23_cv1_act_Mul_output_0[1][128][7][7], const float model_23_m_0_cv1_conv_weight[128][128][1][1], const float model_23_m_0_cv1_conv_bias[128], float _model_23_m_m_0_cv1_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 7, W_in = 7;
    const int K = 128, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_23_m_0_cv1_conv_bias[k];
                float ker_val = model_23_m_0_cv1_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_23_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] = b;
                        _model_23_m_m_0_cv1_conv_Conv_output_0[n][k][h][w] += _model_23_cv1_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_5
    Input: _model_24_Add_1_output_0 [1, 3, 14, 14, 2]
    Input: _model_24_Constant_11_output_0 [1]
    Output: _model_24_Mul_5_output_0 [1, 3, 14, 14, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_5(const float _model_24_Add_1_output_0[1][3][14][14][2], const float _model_24_Constant_11_output_0[1], float _model_24_Mul_5_output_0[1][3][14][14][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_5_output_0[d0][d1][d2][d3][d4] = _model_24_Add_1_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_11_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_7
    Input: _model_24_Pow_1_output_0 [1, 3, 14, 14, 2]
    Input: _model_24_Constant_14_output_0 [1, 3, 14, 14, 2]
    Output: _model_24_Mul_7_output_0 [1, 3, 14, 14, 2]
*/
void node__model_24_Mul_7(const float _model_24_Pow_1_output_0[1][3][14][14][2], const float _model_24_Constant_14_output_0[1][3][14][14][2], float _model_24_Mul_7_output_0[1][3][14][14][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 14; d2++) {
    for (int d3 = 0; d3 < 14; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_7_output_0[d0][d1][d2][d3][d4] = _model_24_Pow_1_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_14_output_0[d0][d1][d2][d3][d4];
    }
    }
    }
    }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.23/m/m.0/cv1/act/Sigmoid
    Input: _model_23_m_m_0_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_23_m_m_0_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_23_m_m_0_cv1_act_Sigmoid(const float _model_23_m_m_0_cv1_conv_Conv_output_0[1][128][7][7], float _model_23_m_m_0_cv1_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_23_m_m_0_cv1_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_23_m_m_0_cv1_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Concat 
    Name in model: /model.24/Concat_1
    Input: _model_24_Mul_5_output_0 [1, 3, 14, 14, 2]
    Input: _model_24_Mul_7_output_0 [1, 3, 14, 14, 2]
    Input: _model_24_Split_1_output_2 [1, 3, 14, 14, 81]
    Output: _model_24_Concat_1_output_0 [1, 3, 14, 14, 85]
*/
void node__model_24_Concat_1(const float _model_24_Mul_5_output_0[1][3][14][14][2], const float _model_24_Mul_7_output_0[1][3][14][14][2], const float _model_24_Split_1_output_2[1][3][14][14][81], float _model_24_Concat_1_output_0[1][3][14][14][85]) {
    /*Concat along axis=4*/
    int axis_offset = 0;
    // Copy tensor '_model_24_Mul_5_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        for (int i4 = 0; i4 < 2; i4++) {
                            _model_24_Concat_1_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Mul_5_output_0[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 2;
    // Copy tensor '_model_24_Mul_7_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        for (int i4 = 0; i4 < 2; i4++) {
                            _model_24_Concat_1_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Mul_7_output_0[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 2;
    // Copy tensor '_model_24_Split_1_output_2' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 14; i2++) {
                    for (int i3 = 0; i3 < 14; i3++) {
                        for (int i4 = 0; i4 < 81; i4++) {
                            _model_24_Concat_1_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Split_1_output_2[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 81;
}

/*  Operator: Mul 
    Name in model: /model.23/m/m.0/cv1/act/Mul
    Input: _model_23_m_m_0_cv1_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_23_m_m_0_cv1_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_23_m_m_0_cv1_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_23_m_m_0_cv1_act_Mul(const float _model_23_m_m_0_cv1_conv_Conv_output_0[1][128][7][7], const float _model_23_m_m_0_cv1_act_Sigmoid_output_0[1][128][7][7], float _model_23_m_m_0_cv1_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_23_m_m_0_cv1_act_Mul_output_0[d0][d1][d2][d3] = _model_23_m_m_0_cv1_conv_Conv_output_0[d0][d1][d2][d3] * _model_23_m_m_0_cv1_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Reshape 
    Name in model: /model.24/Reshape_3
    Input: _model_24_Concat_1_output_0 [1, 3, 14, 14, 85]
    Input: _model_24_Constant_15_output_0 [3]
    Output: _model_24_Reshape_3_output_0 [1, 588, 85]
*/
void node__model_24_Reshape_3(const float _model_24_Concat_1_output_0[1][3][14][14][85], const int64_t _model_24_Constant_15_output_0[3], float _model_24_Reshape_3_output_0[1][588][85]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_model_24_Concat_1_output_0;
    float *dst = (float*)_model_24_Reshape_3_output_0;
    memcpy(dst, src, 49980 * sizeof(float));
}

/*  Operator: Conv 
    Name in model: /model.23/m/m.0/cv2/conv/Conv
    Input: _model_23_m_m_0_cv1_act_Mul_output_0 [1, 128, 7, 7]
    Input: model_23_m_0_cv2_conv_weight [128, 128, 3, 3]
    Input: model_23_m_0_cv2_conv_bias [128]
    Output: _model_23_m_m_0_cv2_conv_Conv_output_0 [1, 128, 7, 7]
*/
void node__model_23_m_m_0_cv2_conv_Conv(const float _model_23_m_m_0_cv1_act_Mul_output_0[1][128][7][7], const float model_23_m_0_cv2_conv_weight[128][128][3][3], const float model_23_m_0_cv2_conv_bias[128], float _model_23_m_m_0_cv2_conv_Conv_output_0[1][128][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 128, H_in = 7, W_in = 7;
    const int K = 128, K_h = 3, K_w = 3;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 1, pad_b = 1, pad_l = 1, pad_r = 1;
    const int dilation_h = 1, dilation_w = 1;
    // General convolution computation using multi-dimensional indexing
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                  sum = model_23_m_0_cv2_conv_bias[k];
                    for (int c = 0; c < C_in; ++c) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int h_in = h * stride_h - pad_t + kh * dilation_h;
                            if (h_in < 0 || h_in >= H_in) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int w_in = w * stride_w - pad_l + kw * dilation_w;
                                if (w_in < 0 || w_in >= W_in) continue;
                                sum += _model_23_m_m_0_cv1_act_Mul_output_0[n][c][h_in][w_in] * model_23_m_0_cv2_conv_weight[k][c][kh][kw];
                            }
                        }
                    }
                    _model_23_m_m_0_cv2_conv_Conv_output_0[n][k][h][w] = sum;
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.23/m/m.0/cv2/act/Sigmoid
    Input: _model_23_m_m_0_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Output: _model_23_m_m_0_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
*/
void node__model_23_m_m_0_cv2_act_Sigmoid(const float _model_23_m_m_0_cv2_conv_Conv_output_0[1][128][7][7], float _model_23_m_m_0_cv2_act_Sigmoid_output_0[1][128][7][7]) {
    float *X_ptr = (float *)_model_23_m_m_0_cv2_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_23_m_m_0_cv2_act_Sigmoid_output_0;
    for (int i = 0; i < 6272; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.23/m/m.0/cv2/act/Mul
    Input: _model_23_m_m_0_cv2_conv_Conv_output_0 [1, 128, 7, 7]
    Input: _model_23_m_m_0_cv2_act_Sigmoid_output_0 [1, 128, 7, 7]
    Output: _model_23_m_m_0_cv2_act_Mul_output_0 [1, 128, 7, 7]
*/
void node__model_23_m_m_0_cv2_act_Mul(const float _model_23_m_m_0_cv2_conv_Conv_output_0[1][128][7][7], const float _model_23_m_m_0_cv2_act_Sigmoid_output_0[1][128][7][7], float _model_23_m_m_0_cv2_act_Mul_output_0[1][128][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 128; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_23_m_m_0_cv2_act_Mul_output_0[d0][d1][d2][d3] = _model_23_m_m_0_cv2_conv_Conv_output_0[d0][d1][d2][d3] * _model_23_m_m_0_cv2_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.23/Concat
    Input: _model_23_m_m_0_cv2_act_Mul_output_0 [1, 128, 7, 7]
    Input: _model_23_cv2_act_Mul_output_0 [1, 128, 7, 7]
    Output: _model_23_Concat_output_0 [1, 256, 7, 7]
*/
void node__model_23_Concat(const float _model_23_m_m_0_cv2_act_Mul_output_0[1][128][7][7], const float _model_23_cv2_act_Mul_output_0[1][128][7][7], float _model_23_Concat_output_0[1][256][7][7]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_23_m_m_0_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_23_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_23_m_m_0_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
    // Copy tensor '_model_23_cv2_act_Mul_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 128; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        _model_23_Concat_output_0[i0][axis_offset + i1][i2][i3] = _model_23_cv2_act_Mul_output_0[i0][i1][i2][i3];
                    }
                }
            }
        }
    }
    axis_offset += 128;
}

/*  Operator: Conv 
    Name in model: /model.23/cv3/conv/Conv
    Input: _model_23_Concat_output_0 [1, 256, 7, 7]
    Input: model_23_cv3_conv_weight [256, 256, 1, 1]
    Input: model_23_cv3_conv_bias [256]
    Output: _model_23_cv3_conv_Conv_output_0 [1, 256, 7, 7]
*/
void node__model_23_cv3_conv_Conv(const float _model_23_Concat_output_0[1][256][7][7], const float model_23_cv3_conv_weight[256][256][1][1], const float model_23_cv3_conv_bias[256], float _model_23_cv3_conv_Conv_output_0[1][256][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 256, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_23_cv3_conv_bias[k];
                float ker_val = model_23_cv3_conv_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_23_cv3_conv_Conv_output_0[n][k][h][w] = b;
                        _model_23_cv3_conv_Conv_output_0[n][k][h][w] += _model_23_Concat_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.23/cv3/act/Sigmoid
    Input: _model_23_cv3_conv_Conv_output_0 [1, 256, 7, 7]
    Output: _model_23_cv3_act_Sigmoid_output_0 [1, 256, 7, 7]
*/
void node__model_23_cv3_act_Sigmoid(const float _model_23_cv3_conv_Conv_output_0[1][256][7][7], float _model_23_cv3_act_Sigmoid_output_0[1][256][7][7]) {
    float *X_ptr = (float *)_model_23_cv3_conv_Conv_output_0;
    float *Y_ptr = (float *)_model_23_cv3_act_Sigmoid_output_0;
    for (int i = 0; i < 12544; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Mul 
    Name in model: /model.23/cv3/act/Mul
    Input: _model_23_cv3_conv_Conv_output_0 [1, 256, 7, 7]
    Input: _model_23_cv3_act_Sigmoid_output_0 [1, 256, 7, 7]
    Output: _model_23_cv3_act_Mul_output_0 [1, 256, 7, 7]
*/
void node__model_23_cv3_act_Mul(const float _model_23_cv3_conv_Conv_output_0[1][256][7][7], const float _model_23_cv3_act_Sigmoid_output_0[1][256][7][7], float _model_23_cv3_act_Mul_output_0[1][256][7][7]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 256; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
        _model_23_cv3_act_Mul_output_0[d0][d1][d2][d3] = _model_23_cv3_conv_Conv_output_0[d0][d1][d2][d3] * _model_23_cv3_act_Sigmoid_output_0[d0][d1][d2][d3];
    }
    }
    }
    }
}

/*  Operator: Conv 
    Name in model: /model.24/m.2/Conv
    Input: _model_23_cv3_act_Mul_output_0 [1, 256, 7, 7]
    Input: model_24_m_2_weight [255, 256, 1, 1]
    Input: model_24_m_2_bias [255]
    Output: _model_24_m_2_Conv_output_0 [1, 255, 7, 7]
*/
void node__model_24_m_2_Conv(const float _model_23_cv3_act_Mul_output_0[1][256][7][7], const float model_24_m_2_weight[255][256][1][1], const float model_24_m_2_bias[255], float _model_24_m_2_Conv_output_0[1][255][7][7]) {
    // Dimension constants
    const int N = 1, C_in = 256, H_in = 7, W_in = 7;
    const int K = 255, K_h = 1, K_w = 1;
    const int H_out = 7, W_out = 7;
    // Convolution parameters
    const int stride_h = 1, stride_w = 1;
    const int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    const int dilation_h = 1, dilation_w = 1;
    // Optimized convolution for 1x1 kernel using multi-dimensional indexing
    // Simplified computation for stride=1, pad=0, dilation=1
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K; ++k) {
                float b= model_24_m_2_bias[k];
                float ker_val = model_24_m_2_weight[k][c][0][0];  // 1x1 kernel
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        /* Accumulate directly into the output */
                    if (c == 0) _model_24_m_2_Conv_output_0[n][k][h][w] = b;
                        _model_24_m_2_Conv_output_0[n][k][h][w] += _model_23_cv3_act_Mul_output_0[n][c][h][w] * ker_val;
                    }
                }
            }
        }
    }
}

/*  Operator: Reshape 
    Name in model: /model.24/Reshape_4
    Input: _model_24_m_2_Conv_output_0 [1, 255, 7, 7]
    Input: _model_24_Constant_16_output_0 [5]
    Output: _model_24_Reshape_4_output_0 [1, 3, 85, 7, 7]
*/
void node__model_24_Reshape_4(const float _model_24_m_2_Conv_output_0[1][255][7][7], const int64_t _model_24_Constant_16_output_0[5], float _model_24_Reshape_4_output_0[1][3][85][7][7]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_model_24_m_2_Conv_output_0;
    float *dst = (float*)_model_24_Reshape_4_output_0;
    memcpy(dst, src, 12495 * sizeof(float));
}

/*  Operator: Transpose 
    Name in model: /model.24/Transpose_2
    Input: _model_24_Reshape_4_output_0 [1, 3, 85, 7, 7]
    Output: _model_24_Transpose_2_output_0 [1, 3, 7, 7, 85]
*/
void node__model_24_Transpose_2(const float _model_24_Reshape_4_output_0[1][3][85][7][7], float _model_24_Transpose_2_output_0[1][3][7][7][85]) {
    /*Transpose with perm = [0, 1, 3, 4, 2]*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 7; i2++) {
                for (int i3 = 0; i3 < 7; i3++) {
                    for (int i4 = 0; i4 < 85; i4++) {
                        _model_24_Transpose_2_output_0[i0][i1][i2][i3][i4] = _model_24_Reshape_4_output_0[i0][i1][i4][i2][i3];
                    }
                }
            }
        }
    }
}

/*  Operator: Sigmoid 
    Name in model: /model.24/Sigmoid_2
    Input: _model_24_Transpose_2_output_0 [1, 3, 7, 7, 85]
    Output: _model_24_Sigmoid_2_output_0 [1, 3, 7, 7, 85]
*/
void node__model_24_Sigmoid_2(const float _model_24_Transpose_2_output_0[1][3][7][7][85], float _model_24_Sigmoid_2_output_0[1][3][7][7][85]) {
    float *X_ptr = (float *)_model_24_Transpose_2_output_0;
    float *Y_ptr = (float *)_model_24_Sigmoid_2_output_0;
    for (int i = 0; i < 12495; i++) {
        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));
    }
}

/*  Operator: Split 
    Name in model: /model.24/Split_2
    Input: _model_24_Sigmoid_2_output_0 [1, 3, 7, 7, 85]
    Input: onnx__Split_377 [3]
    Output: _model_24_Split_2_output_0 [1, 3, 7, 7, 2]    Output: _model_24_Split_2_output_1 [1, 3, 7, 7, 2]    Output: _model_24_Split_2_output_2 [1, 3, 7, 7, 81]
*/
void node__model_24_Split_2(const float _model_24_Sigmoid_2_output_0[1][3][7][7][85], const int64_t onnx__Split_377[3], float _model_24_Split_2_output_0[1][3][7][7][2], float _model_24_Split_2_output_1[1][3][7][7][2], float _model_24_Split_2_output_2[1][3][7][7][81]) {
    // Split along axis=4
    const int64_t* split = (const int64_t*)onnx__Split_377;
    // Processing output 0: _model_24_Split_2_output_0
    const int64_t start_0 = 0;
    const int64_t split_size_0 = split[0];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 7; i2++) {
                for (int i3 = 0; i3 < 7; i3++) {
                    for (int i4 = 0; i4 < split_size_0; i4++) {
                        _model_24_Split_2_output_0[i0][i1][i2][i3][i4] = _model_24_Sigmoid_2_output_0[i0][i1][i2][i3][start_0 + i4];
                    }
                }
            }
        }
    }
    // Processing output 1: _model_24_Split_2_output_1
    const int64_t start_1 = split[0];
    const int64_t split_size_1 = split[1];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 7; i2++) {
                for (int i3 = 0; i3 < 7; i3++) {
                    for (int i4 = 0; i4 < split_size_1; i4++) {
                        _model_24_Split_2_output_1[i0][i1][i2][i3][i4] = _model_24_Sigmoid_2_output_0[i0][i1][i2][i3][start_1 + i4];
                    }
                }
            }
        }
    }
    // Processing output 2: _model_24_Split_2_output_2
    const int64_t start_2 = split[0] + split[1];
    const int64_t split_size_2 = split[2];
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 7; i2++) {
                for (int i3 = 0; i3 < 7; i3++) {
                    for (int i4 = 0; i4 < split_size_2; i4++) {
                        _model_24_Split_2_output_2[i0][i1][i2][i3][i4] = _model_24_Sigmoid_2_output_0[i0][i1][i2][i3][start_2 + i4];
                    }
                }
            }
        }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_8
    Input: _model_24_Split_2_output_0 [1, 3, 7, 7, 2]
    Input: _model_24_Constant_1_output_0 [1]
    Output: _model_24_Mul_8_output_0 [1, 3, 7, 7, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_8(const float _model_24_Split_2_output_0[1][3][7][7][2], const float _model_24_Constant_1_output_0[1], float _model_24_Mul_8_output_0[1][3][7][7][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_8_output_0[d0][d1][d2][d3][d4] = _model_24_Split_2_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_1_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_10
    Input: _model_24_Split_2_output_1 [1, 3, 7, 7, 2]
    Input: _model_24_Constant_1_output_0 [1]
    Output: _model_24_Mul_10_output_0 [1, 3, 7, 7, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_10(const float _model_24_Split_2_output_1[1][3][7][7][2], const float _model_24_Constant_1_output_0[1], float _model_24_Mul_10_output_0[1][3][7][7][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_10_output_0[d0][d1][d2][d3][d4] = _model_24_Split_2_output_1[d0][d1][d2][d3][d4] * _model_24_Constant_1_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /model.24/Add_2
    Input: _model_24_Mul_8_output_0 [1, 3, 7, 7, 2]
    Input: _model_24_Constant_18_output_0 [1, 3, 7, 7, 2]
    Output: _model_24_Add_2_output_0 [1, 3, 7, 7, 2]
*/
void node__model_24_Add_2(const float _model_24_Mul_8_output_0[1][3][7][7][2], const float _model_24_Constant_18_output_0[1][3][7][7][2], float _model_24_Add_2_output_0[1][3][7][7][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Add_2_output_0[d0][d1][d2][d3][d4] = _model_24_Mul_8_output_0[d0][d1][d2][d3][d4] + _model_24_Constant_18_output_0[d0][d1][d2][d3][d4];
    }
    }
    }
    }
    }
}

/*  Operator: Pow 
    Name in model: /model.24/Pow_2
    Input: _model_24_Mul_10_output_0 [1, 3, 7, 7, 2]
    Input: _model_24_Constant_5_output_0 [1]
    Output: _model_24_Pow_2_output_0 [1, 3, 7, 7, 2]
*/
void node__model_24_Pow_2(const float _model_24_Mul_10_output_0[1][3][7][7][2], const float _model_24_Constant_5_output_0[1], float _model_24_Pow_2_output_0[1][3][7][7][2]) {
    /*pow*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 7; i2++) {
                for (int i3 = 0; i3 < 7; i3++) {
                    for (int i4 = 0; i4 < 2; i4++) {
                        _model_24_Pow_2_output_0[i0][i1][i2][i3][i4] = pow(_model_24_Mul_10_output_0[i0][i1][i2][i3][i4], _model_24_Constant_5_output_0[0]);
                    }
                }
            }
        }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_9
    Input: _model_24_Add_2_output_0 [1, 3, 7, 7, 2]
    Input: _model_24_Constant_19_output_0 [1]
    Output: _model_24_Mul_9_output_0 [1, 3, 7, 7, 2]
*/
    // Warning: Broadcasting is applied.
void node__model_24_Mul_9(const float _model_24_Add_2_output_0[1][3][7][7][2], const float _model_24_Constant_19_output_0[1], float _model_24_Mul_9_output_0[1][3][7][7][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_9_output_0[d0][d1][d2][d3][d4] = _model_24_Add_2_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_19_output_0[0];
    }
    }
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /model.24/Mul_11
    Input: _model_24_Pow_2_output_0 [1, 3, 7, 7, 2]
    Input: _model_24_Constant_22_output_0 [1, 3, 7, 7, 2]
    Output: _model_24_Mul_11_output_0 [1, 3, 7, 7, 2]
*/
void node__model_24_Mul_11(const float _model_24_Pow_2_output_0[1][3][7][7][2], const float _model_24_Constant_22_output_0[1][3][7][7][2], float _model_24_Mul_11_output_0[1][3][7][7][2]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 3; d1++) {
    for (int d2 = 0; d2 < 7; d2++) {
    for (int d3 = 0; d3 < 7; d3++) {
    for (int d4 = 0; d4 < 2; d4++) {
        _model_24_Mul_11_output_0[d0][d1][d2][d3][d4] = _model_24_Pow_2_output_0[d0][d1][d2][d3][d4] * _model_24_Constant_22_output_0[d0][d1][d2][d3][d4];
    }
    }
    }
    }
    }
}

/*  Operator: Concat 
    Name in model: /model.24/Concat_2
    Input: _model_24_Mul_9_output_0 [1, 3, 7, 7, 2]
    Input: _model_24_Mul_11_output_0 [1, 3, 7, 7, 2]
    Input: _model_24_Split_2_output_2 [1, 3, 7, 7, 81]
    Output: _model_24_Concat_2_output_0 [1, 3, 7, 7, 85]
*/
void node__model_24_Concat_2(const float _model_24_Mul_9_output_0[1][3][7][7][2], const float _model_24_Mul_11_output_0[1][3][7][7][2], const float _model_24_Split_2_output_2[1][3][7][7][81], float _model_24_Concat_2_output_0[1][3][7][7][85]) {
    /*Concat along axis=4*/
    int axis_offset = 0;
    // Copy tensor '_model_24_Mul_9_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        for (int i4 = 0; i4 < 2; i4++) {
                            _model_24_Concat_2_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Mul_9_output_0[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 2;
    // Copy tensor '_model_24_Mul_11_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        for (int i4 = 0; i4 < 2; i4++) {
                            _model_24_Concat_2_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Mul_11_output_0[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 2;
    // Copy tensor '_model_24_Split_2_output_2' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i2 = 0; i2 < 7; i2++) {
                    for (int i3 = 0; i3 < 7; i3++) {
                        for (int i4 = 0; i4 < 81; i4++) {
                            _model_24_Concat_2_output_0[i0][i1][i2][i3][axis_offset + i4] = _model_24_Split_2_output_2[i0][i1][i2][i3][i4];
                        }
                    }
                }
            }
        }
    }
    axis_offset += 81;
}

/*  Operator: Reshape 
    Name in model: /model.24/Reshape_5
    Input: _model_24_Concat_2_output_0 [1, 3, 7, 7, 85]
    Input: _model_24_Constant_23_output_0 [3]
    Output: _model_24_Reshape_5_output_0 [1, 147, 85]
*/
void node__model_24_Reshape_5(const float _model_24_Concat_2_output_0[1][3][7][7][85], const int64_t _model_24_Constant_23_output_0[3], float _model_24_Reshape_5_output_0[1][147][85]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_model_24_Concat_2_output_0;
    float *dst = (float*)_model_24_Reshape_5_output_0;
    memcpy(dst, src, 12495 * sizeof(float));
}

/*  Operator: Concat 
    Name in model: /model.24/Concat_3
    Input: _model_24_Reshape_1_output_0 [1, 2352, 85]
    Input: _model_24_Reshape_3_output_0 [1, 588, 85]
    Input: _model_24_Reshape_5_output_0 [1, 147, 85]
    Output: output0 [1, 3087, 85]
*/
void node__model_24_Concat_3(const float _model_24_Reshape_1_output_0[1][2352][85], const float _model_24_Reshape_3_output_0[1][588][85], const float _model_24_Reshape_5_output_0[1][147][85], float output0[1][3087][85]) {
    /*Concat along axis=1*/
    int axis_offset = 0;
    // Copy tensor '_model_24_Reshape_1_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 2352; i1++) {
                for (int i2 = 0; i2 < 85; i2++) {
                    output0[i0][axis_offset + i1][i2] = _model_24_Reshape_1_output_0[i0][i1][i2];
                }
            }
        }
    }
    axis_offset += 2352;
    // Copy tensor '_model_24_Reshape_3_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 588; i1++) {
                for (int i2 = 0; i2 < 85; i2++) {
                    output0[i0][axis_offset + i1][i2] = _model_24_Reshape_3_output_0[i0][i1][i2];
                }
            }
        }
    }
    axis_offset += 588;
    // Copy tensor '_model_24_Reshape_5_output_0' into the output
    {
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 147; i1++) {
                for (int i2 = 0; i2 < 85; i2++) {
                    output0[i0][axis_offset + i1][i2] = _model_24_Reshape_5_output_0[i0][i1][i2];
                }
            }
        }
    }
    axis_offset += 147;
}

void forward_pass(const float images[1][3][224][224], float output0[1][3087][85])
{
      union tensor_union_0 *tu0 = (union tensor_union_0 *)ps_malloc(sizeof(union tensor_union_0));
      union tensor_union_1 *tu1 = (union tensor_union_1 *)ps_malloc(sizeof(union tensor_union_1));
      union tensor_union_2 *tu2 = (union tensor_union_2 *)ps_malloc(sizeof(union tensor_union_2));
      union tensor_union_3 *tu3 = (union tensor_union_3 *)ps_malloc(sizeof(union tensor_union_3));
      union tensor_union_4 *tu4 = (union tensor_union_4 *)ps_malloc(sizeof(union tensor_union_4));
      union tensor_union_5 *tu5 = (union tensor_union_5 *)ps_malloc(sizeof(union tensor_union_5));
      union tensor_union_6 *tu6 = (union tensor_union_6 *)ps_malloc(sizeof(union tensor_union_6));
      union tensor_union_7 *tu7 = (union tensor_union_7 *)ps_malloc(sizeof(union tensor_union_7));
      union tensor_union_8 *tu8 = (union tensor_union_8 *)ps_malloc(sizeof(union tensor_union_8));
    node__model_0_conv_Conv(images, tensor_model_0_conv_weight, tensor_model_0_conv_bias, tu0->tensor__model_0_conv_Conv_Output_0);
    node__model_0_act_Sigmoid(tu0->tensor__model_0_conv_Conv_Output_0, tu1->tensor__model_0_act_Sigmoid_Output_0);
    node__model_0_act_Mul(tu0->tensor__model_0_conv_Conv_Output_0, tu1->tensor__model_0_act_Sigmoid_Output_0, tu2->tensor__model_0_act_Mul_Output_0);
    node__model_1_conv_Conv(tu2->tensor__model_0_act_Mul_Output_0, tensor_model_1_conv_weight, tensor_model_1_conv_bias, tu0->tensor__model_1_conv_Conv_Output_0);
    node__model_1_act_Sigmoid(tu0->tensor__model_1_conv_Conv_Output_0, tu1->tensor__model_1_act_Sigmoid_Output_0);
    node__model_1_act_Mul(tu0->tensor__model_1_conv_Conv_Output_0, tu1->tensor__model_1_act_Sigmoid_Output_0, tu2->tensor__model_1_act_Mul_Output_0);
    node__model_2_cv1_conv_Conv(tu2->tensor__model_1_act_Mul_Output_0, tensor_model_2_cv1_conv_weight, tensor_model_2_cv1_conv_bias, tu0->tensor__model_2_cv1_conv_Conv_Output_0);
    node__model_2_cv2_conv_Conv(tu2->tensor__model_1_act_Mul_Output_0, tensor_model_2_cv2_conv_weight, tensor_model_2_cv2_conv_bias, tu1->tensor__model_2_cv2_conv_Conv_Output_0);
    node__model_2_cv1_act_Sigmoid(tu0->tensor__model_2_cv1_conv_Conv_Output_0, tu2->tensor__model_2_cv1_act_Sigmoid_Output_0);
    node__model_2_cv2_act_Sigmoid(tu1->tensor__model_2_cv2_conv_Conv_Output_0, tu3->tensor__model_2_cv2_act_Sigmoid_Output_0);
    node__model_2_cv1_act_Mul(tu0->tensor__model_2_cv1_conv_Conv_Output_0, tu2->tensor__model_2_cv1_act_Sigmoid_Output_0, tu4->tensor__model_2_cv1_act_Mul_Output_0);
    node__model_2_cv2_act_Mul(tu1->tensor__model_2_cv2_conv_Conv_Output_0, tu3->tensor__model_2_cv2_act_Sigmoid_Output_0, tu0->tensor__model_2_cv2_act_Mul_Output_0);
    node__model_2_m_m_0_cv1_conv_Conv(tu4->tensor__model_2_cv1_act_Mul_Output_0, tensor_model_2_m_0_cv1_conv_weight, tensor_model_2_m_0_cv1_conv_bias, tu1->tensor__model_2_m_m_0_cv1_conv_Conv_Output_0);
    node__model_2_m_m_0_cv1_act_Sigmoid(tu1->tensor__model_2_m_m_0_cv1_conv_Conv_Output_0, tu2->tensor__model_2_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_2_m_m_0_cv1_act_Mul(tu1->tensor__model_2_m_m_0_cv1_conv_Conv_Output_0, tu2->tensor__model_2_m_m_0_cv1_act_Sigmoid_Output_0, tu3->tensor__model_2_m_m_0_cv1_act_Mul_Output_0);
    node__model_2_m_m_0_cv2_conv_Conv(tu3->tensor__model_2_m_m_0_cv1_act_Mul_Output_0, tensor_model_2_m_0_cv2_conv_weight, tensor_model_2_m_0_cv2_conv_bias, tu1->tensor__model_2_m_m_0_cv2_conv_Conv_Output_0);
    node__model_2_m_m_0_cv2_act_Sigmoid(tu1->tensor__model_2_m_m_0_cv2_conv_Conv_Output_0, tu2->tensor__model_2_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_2_m_m_0_cv2_act_Mul(tu1->tensor__model_2_m_m_0_cv2_conv_Conv_Output_0, tu2->tensor__model_2_m_m_0_cv2_act_Sigmoid_Output_0, tu3->tensor__model_2_m_m_0_cv2_act_Mul_Output_0);
    node__model_2_m_m_0_Add(tu4->tensor__model_2_cv1_act_Mul_Output_0, tu3->tensor__model_2_m_m_0_cv2_act_Mul_Output_0, tu1->tensor__model_2_m_m_0_Add_Output_0);
    node__model_2_Concat(tu1->tensor__model_2_m_m_0_Add_Output_0, tu0->tensor__model_2_cv2_act_Mul_Output_0, tu2->tensor__model_2_Concat_Output_0);
    node__model_2_cv3_conv_Conv(tu2->tensor__model_2_Concat_Output_0, tensor_model_2_cv3_conv_weight, tensor_model_2_cv3_conv_bias, tu0->tensor__model_2_cv3_conv_Conv_Output_0);
    node__model_2_cv3_act_Sigmoid(tu0->tensor__model_2_cv3_conv_Conv_Output_0, tu1->tensor__model_2_cv3_act_Sigmoid_Output_0);
    node__model_2_cv3_act_Mul(tu0->tensor__model_2_cv3_conv_Conv_Output_0, tu1->tensor__model_2_cv3_act_Sigmoid_Output_0, tu2->tensor__model_2_cv3_act_Mul_Output_0);
    node__model_3_conv_Conv(tu2->tensor__model_2_cv3_act_Mul_Output_0, tensor_model_3_conv_weight, tensor_model_3_conv_bias, tu0->tensor__model_3_conv_Conv_Output_0);
    node__model_3_act_Sigmoid(tu0->tensor__model_3_conv_Conv_Output_0, tu1->tensor__model_3_act_Sigmoid_Output_0);
    node__model_3_act_Mul(tu0->tensor__model_3_conv_Conv_Output_0, tu1->tensor__model_3_act_Sigmoid_Output_0, tu2->tensor__model_3_act_Mul_Output_0);
    node__model_4_cv1_conv_Conv(tu2->tensor__model_3_act_Mul_Output_0, tensor_model_4_cv1_conv_weight, tensor_model_4_cv1_conv_bias, tu0->tensor__model_4_cv1_conv_Conv_Output_0);
    node__model_4_cv2_conv_Conv(tu2->tensor__model_3_act_Mul_Output_0, tensor_model_4_cv2_conv_weight, tensor_model_4_cv2_conv_bias, tu1->tensor__model_4_cv2_conv_Conv_Output_0);
    node__model_4_cv1_act_Sigmoid(tu0->tensor__model_4_cv1_conv_Conv_Output_0, tu2->tensor__model_4_cv1_act_Sigmoid_Output_0);
    node__model_4_cv2_act_Sigmoid(tu1->tensor__model_4_cv2_conv_Conv_Output_0, tu3->tensor__model_4_cv2_act_Sigmoid_Output_0);
    node__model_4_cv1_act_Mul(tu0->tensor__model_4_cv1_conv_Conv_Output_0, tu2->tensor__model_4_cv1_act_Sigmoid_Output_0, tu4->tensor__model_4_cv1_act_Mul_Output_0);
    node__model_4_cv2_act_Mul(tu1->tensor__model_4_cv2_conv_Conv_Output_0, tu3->tensor__model_4_cv2_act_Sigmoid_Output_0, tu0->tensor__model_4_cv2_act_Mul_Output_0);
    node__model_4_m_m_0_cv1_conv_Conv(tu4->tensor__model_4_cv1_act_Mul_Output_0, tensor_model_4_m_0_cv1_conv_weight, tensor_model_4_m_0_cv1_conv_bias, tu1->tensor__model_4_m_m_0_cv1_conv_Conv_Output_0);
    node__model_4_m_m_0_cv1_act_Sigmoid(tu1->tensor__model_4_m_m_0_cv1_conv_Conv_Output_0, tu2->tensor__model_4_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_4_m_m_0_cv1_act_Mul(tu1->tensor__model_4_m_m_0_cv1_conv_Conv_Output_0, tu2->tensor__model_4_m_m_0_cv1_act_Sigmoid_Output_0, tu3->tensor__model_4_m_m_0_cv1_act_Mul_Output_0);
    node__model_4_m_m_0_cv2_conv_Conv(tu3->tensor__model_4_m_m_0_cv1_act_Mul_Output_0, tensor_model_4_m_0_cv2_conv_weight, tensor_model_4_m_0_cv2_conv_bias, tu1->tensor__model_4_m_m_0_cv2_conv_Conv_Output_0);
    node__model_4_m_m_0_cv2_act_Sigmoid(tu1->tensor__model_4_m_m_0_cv2_conv_Conv_Output_0, tu2->tensor__model_4_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_4_m_m_0_cv2_act_Mul(tu1->tensor__model_4_m_m_0_cv2_conv_Conv_Output_0, tu2->tensor__model_4_m_m_0_cv2_act_Sigmoid_Output_0, tu3->tensor__model_4_m_m_0_cv2_act_Mul_Output_0);
    node__model_4_m_m_0_Add(tu4->tensor__model_4_cv1_act_Mul_Output_0, tu3->tensor__model_4_m_m_0_cv2_act_Mul_Output_0, tu1->tensor__model_4_m_m_0_Add_Output_0);
    node__model_4_m_m_1_cv1_conv_Conv(tu1->tensor__model_4_m_m_0_Add_Output_0, tensor_model_4_m_1_cv1_conv_weight, tensor_model_4_m_1_cv1_conv_bias, tu2->tensor__model_4_m_m_1_cv1_conv_Conv_Output_0);
    node__model_4_m_m_1_cv1_act_Sigmoid(tu2->tensor__model_4_m_m_1_cv1_conv_Conv_Output_0, tu3->tensor__model_4_m_m_1_cv1_act_Sigmoid_Output_0);
    node__model_4_m_m_1_cv1_act_Mul(tu2->tensor__model_4_m_m_1_cv1_conv_Conv_Output_0, tu3->tensor__model_4_m_m_1_cv1_act_Sigmoid_Output_0, tu4->tensor__model_4_m_m_1_cv1_act_Mul_Output_0);
    node__model_4_m_m_1_cv2_conv_Conv(tu4->tensor__model_4_m_m_1_cv1_act_Mul_Output_0, tensor_model_4_m_1_cv2_conv_weight, tensor_model_4_m_1_cv2_conv_bias, tu2->tensor__model_4_m_m_1_cv2_conv_Conv_Output_0);
    node__model_4_m_m_1_cv2_act_Sigmoid(tu2->tensor__model_4_m_m_1_cv2_conv_Conv_Output_0, tu3->tensor__model_4_m_m_1_cv2_act_Sigmoid_Output_0);
    node__model_4_m_m_1_cv2_act_Mul(tu2->tensor__model_4_m_m_1_cv2_conv_Conv_Output_0, tu3->tensor__model_4_m_m_1_cv2_act_Sigmoid_Output_0, tu4->tensor__model_4_m_m_1_cv2_act_Mul_Output_0);
    node__model_4_m_m_1_Add(tu1->tensor__model_4_m_m_0_Add_Output_0, tu4->tensor__model_4_m_m_1_cv2_act_Mul_Output_0, tu2->tensor__model_4_m_m_1_Add_Output_0);
    node__model_4_Concat(tu2->tensor__model_4_m_m_1_Add_Output_0, tu0->tensor__model_4_cv2_act_Mul_Output_0, tu1->tensor__model_4_Concat_Output_0);
    node__model_4_cv3_conv_Conv(tu1->tensor__model_4_Concat_Output_0, tensor_model_4_cv3_conv_weight, tensor_model_4_cv3_conv_bias, tu0->tensor__model_4_cv3_conv_Conv_Output_0);
    node__model_4_cv3_act_Sigmoid(tu0->tensor__model_4_cv3_conv_Conv_Output_0, tu1->tensor__model_4_cv3_act_Sigmoid_Output_0);
    node__model_4_cv3_act_Mul(tu0->tensor__model_4_cv3_conv_Conv_Output_0, tu1->tensor__model_4_cv3_act_Sigmoid_Output_0, tu2->tensor__model_4_cv3_act_Mul_Output_0);
    node__model_5_conv_Conv(tu2->tensor__model_4_cv3_act_Mul_Output_0, tensor_model_5_conv_weight, tensor_model_5_conv_bias, tu0->tensor__model_5_conv_Conv_Output_0);
    node__model_5_act_Sigmoid(tu0->tensor__model_5_conv_Conv_Output_0, tu1->tensor__model_5_act_Sigmoid_Output_0);
    node__model_5_act_Mul(tu0->tensor__model_5_conv_Conv_Output_0, tu1->tensor__model_5_act_Sigmoid_Output_0, tu3->tensor__model_5_act_Mul_Output_0);
    node__model_6_cv1_conv_Conv(tu3->tensor__model_5_act_Mul_Output_0, tensor_model_6_cv1_conv_weight, tensor_model_6_cv1_conv_bias, tu0->tensor__model_6_cv1_conv_Conv_Output_0);
    node__model_6_cv2_conv_Conv(tu3->tensor__model_5_act_Mul_Output_0, tensor_model_6_cv2_conv_weight, tensor_model_6_cv2_conv_bias, tu1->tensor__model_6_cv2_conv_Conv_Output_0);
    node__model_6_cv1_act_Sigmoid(tu0->tensor__model_6_cv1_conv_Conv_Output_0, tu3->tensor__model_6_cv1_act_Sigmoid_Output_0);
    node__model_6_cv2_act_Sigmoid(tu1->tensor__model_6_cv2_conv_Conv_Output_0, tu4->tensor__model_6_cv2_act_Sigmoid_Output_0);
    node__model_6_cv1_act_Mul(tu0->tensor__model_6_cv1_conv_Conv_Output_0, tu3->tensor__model_6_cv1_act_Sigmoid_Output_0, tu5->tensor__model_6_cv1_act_Mul_Output_0);
    node__model_6_cv2_act_Mul(tu1->tensor__model_6_cv2_conv_Conv_Output_0, tu4->tensor__model_6_cv2_act_Sigmoid_Output_0, tu0->tensor__model_6_cv2_act_Mul_Output_0);
    node__model_6_m_m_0_cv1_conv_Conv(tu5->tensor__model_6_cv1_act_Mul_Output_0, tensor_model_6_m_0_cv1_conv_weight, tensor_model_6_m_0_cv1_conv_bias, tu1->tensor__model_6_m_m_0_cv1_conv_Conv_Output_0);
    node__model_6_m_m_0_cv1_act_Sigmoid(tu1->tensor__model_6_m_m_0_cv1_conv_Conv_Output_0, tu3->tensor__model_6_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_6_m_m_0_cv1_act_Mul(tu1->tensor__model_6_m_m_0_cv1_conv_Conv_Output_0, tu3->tensor__model_6_m_m_0_cv1_act_Sigmoid_Output_0, tu4->tensor__model_6_m_m_0_cv1_act_Mul_Output_0);
    node__model_6_m_m_0_cv2_conv_Conv(tu4->tensor__model_6_m_m_0_cv1_act_Mul_Output_0, tensor_model_6_m_0_cv2_conv_weight, tensor_model_6_m_0_cv2_conv_bias, tu1->tensor__model_6_m_m_0_cv2_conv_Conv_Output_0);
    node__model_6_m_m_0_cv2_act_Sigmoid(tu1->tensor__model_6_m_m_0_cv2_conv_Conv_Output_0, tu3->tensor__model_6_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_6_m_m_0_cv2_act_Mul(tu1->tensor__model_6_m_m_0_cv2_conv_Conv_Output_0, tu3->tensor__model_6_m_m_0_cv2_act_Sigmoid_Output_0, tu4->tensor__model_6_m_m_0_cv2_act_Mul_Output_0);
    node__model_6_m_m_0_Add(tu5->tensor__model_6_cv1_act_Mul_Output_0, tu4->tensor__model_6_m_m_0_cv2_act_Mul_Output_0, tu1->tensor__model_6_m_m_0_Add_Output_0);
    node__model_6_m_m_1_cv1_conv_Conv(tu1->tensor__model_6_m_m_0_Add_Output_0, tensor_model_6_m_1_cv1_conv_weight, tensor_model_6_m_1_cv1_conv_bias, tu3->tensor__model_6_m_m_1_cv1_conv_Conv_Output_0);
    node__model_6_m_m_1_cv1_act_Sigmoid(tu3->tensor__model_6_m_m_1_cv1_conv_Conv_Output_0, tu4->tensor__model_6_m_m_1_cv1_act_Sigmoid_Output_0);
    node__model_6_m_m_1_cv1_act_Mul(tu3->tensor__model_6_m_m_1_cv1_conv_Conv_Output_0, tu4->tensor__model_6_m_m_1_cv1_act_Sigmoid_Output_0, tu5->tensor__model_6_m_m_1_cv1_act_Mul_Output_0);
    node__model_6_m_m_1_cv2_conv_Conv(tu5->tensor__model_6_m_m_1_cv1_act_Mul_Output_0, tensor_model_6_m_1_cv2_conv_weight, tensor_model_6_m_1_cv2_conv_bias, tu3->tensor__model_6_m_m_1_cv2_conv_Conv_Output_0);
    node__model_6_m_m_1_cv2_act_Sigmoid(tu3->tensor__model_6_m_m_1_cv2_conv_Conv_Output_0, tu4->tensor__model_6_m_m_1_cv2_act_Sigmoid_Output_0);
    node__model_6_m_m_1_cv2_act_Mul(tu3->tensor__model_6_m_m_1_cv2_conv_Conv_Output_0, tu4->tensor__model_6_m_m_1_cv2_act_Sigmoid_Output_0, tu5->tensor__model_6_m_m_1_cv2_act_Mul_Output_0);
    node__model_6_m_m_1_Add(tu1->tensor__model_6_m_m_0_Add_Output_0, tu5->tensor__model_6_m_m_1_cv2_act_Mul_Output_0, tu3->tensor__model_6_m_m_1_Add_Output_0);
    node__model_6_m_m_2_cv1_conv_Conv(tu3->tensor__model_6_m_m_1_Add_Output_0, tensor_model_6_m_2_cv1_conv_weight, tensor_model_6_m_2_cv1_conv_bias, tu1->tensor__model_6_m_m_2_cv1_conv_Conv_Output_0);
    node__model_6_m_m_2_cv1_act_Sigmoid(tu1->tensor__model_6_m_m_2_cv1_conv_Conv_Output_0, tu4->tensor__model_6_m_m_2_cv1_act_Sigmoid_Output_0);
    node__model_6_m_m_2_cv1_act_Mul(tu1->tensor__model_6_m_m_2_cv1_conv_Conv_Output_0, tu4->tensor__model_6_m_m_2_cv1_act_Sigmoid_Output_0, tu5->tensor__model_6_m_m_2_cv1_act_Mul_Output_0);
    node__model_6_m_m_2_cv2_conv_Conv(tu5->tensor__model_6_m_m_2_cv1_act_Mul_Output_0, tensor_model_6_m_2_cv2_conv_weight, tensor_model_6_m_2_cv2_conv_bias, tu1->tensor__model_6_m_m_2_cv2_conv_Conv_Output_0);
    node__model_6_m_m_2_cv2_act_Sigmoid(tu1->tensor__model_6_m_m_2_cv2_conv_Conv_Output_0, tu4->tensor__model_6_m_m_2_cv2_act_Sigmoid_Output_0);
    node__model_6_m_m_2_cv2_act_Mul(tu1->tensor__model_6_m_m_2_cv2_conv_Conv_Output_0, tu4->tensor__model_6_m_m_2_cv2_act_Sigmoid_Output_0, tu5->tensor__model_6_m_m_2_cv2_act_Mul_Output_0);
    node__model_6_m_m_2_Add(tu3->tensor__model_6_m_m_1_Add_Output_0, tu5->tensor__model_6_m_m_2_cv2_act_Mul_Output_0, tu1->tensor__model_6_m_m_2_Add_Output_0);
    node__model_6_Concat(tu1->tensor__model_6_m_m_2_Add_Output_0, tu0->tensor__model_6_cv2_act_Mul_Output_0, tu3->tensor__model_6_Concat_Output_0);
    node__model_6_cv3_conv_Conv(tu3->tensor__model_6_Concat_Output_0, tensor_model_6_cv3_conv_weight, tensor_model_6_cv3_conv_bias, tu0->tensor__model_6_cv3_conv_Conv_Output_0);
    node__model_6_cv3_act_Sigmoid(tu0->tensor__model_6_cv3_conv_Conv_Output_0, tu1->tensor__model_6_cv3_act_Sigmoid_Output_0);
    node__model_6_cv3_act_Mul(tu0->tensor__model_6_cv3_conv_Conv_Output_0, tu1->tensor__model_6_cv3_act_Sigmoid_Output_0, tu3->tensor__model_6_cv3_act_Mul_Output_0);
    node__model_7_conv_Conv(tu3->tensor__model_6_cv3_act_Mul_Output_0, tensor_model_7_conv_weight, tensor_model_7_conv_bias, tu0->tensor__model_7_conv_Conv_Output_0);
    node__model_7_act_Sigmoid(tu0->tensor__model_7_conv_Conv_Output_0, tu1->tensor__model_7_act_Sigmoid_Output_0);
    node__model_7_act_Mul(tu0->tensor__model_7_conv_Conv_Output_0, tu1->tensor__model_7_act_Sigmoid_Output_0, tu4->tensor__model_7_act_Mul_Output_0);
    node__model_8_cv1_conv_Conv(tu4->tensor__model_7_act_Mul_Output_0, tensor_model_8_cv1_conv_weight, tensor_model_8_cv1_conv_bias, tu0->tensor__model_8_cv1_conv_Conv_Output_0);
    node__model_8_cv2_conv_Conv(tu4->tensor__model_7_act_Mul_Output_0, tensor_model_8_cv2_conv_weight, tensor_model_8_cv2_conv_bias, tu1->tensor__model_8_cv2_conv_Conv_Output_0);
    node__model_8_cv1_act_Sigmoid(tu0->tensor__model_8_cv1_conv_Conv_Output_0, tu4->tensor__model_8_cv1_act_Sigmoid_Output_0);
    node__model_8_cv2_act_Sigmoid(tu1->tensor__model_8_cv2_conv_Conv_Output_0, tu5->tensor__model_8_cv2_act_Sigmoid_Output_0);
    node__model_8_cv1_act_Mul(tu0->tensor__model_8_cv1_conv_Conv_Output_0, tu4->tensor__model_8_cv1_act_Sigmoid_Output_0, tu6->tensor__model_8_cv1_act_Mul_Output_0);
    node__model_8_cv2_act_Mul(tu1->tensor__model_8_cv2_conv_Conv_Output_0, tu5->tensor__model_8_cv2_act_Sigmoid_Output_0, tu0->tensor__model_8_cv2_act_Mul_Output_0);
    node__model_8_m_m_0_cv1_conv_Conv(tu6->tensor__model_8_cv1_act_Mul_Output_0, tensor_model_8_m_0_cv1_conv_weight, tensor_model_8_m_0_cv1_conv_bias, tu1->tensor__model_8_m_m_0_cv1_conv_Conv_Output_0);
    node__model_8_m_m_0_cv1_act_Sigmoid(tu1->tensor__model_8_m_m_0_cv1_conv_Conv_Output_0, tu4->tensor__model_8_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_8_m_m_0_cv1_act_Mul(tu1->tensor__model_8_m_m_0_cv1_conv_Conv_Output_0, tu4->tensor__model_8_m_m_0_cv1_act_Sigmoid_Output_0, tu5->tensor__model_8_m_m_0_cv1_act_Mul_Output_0);
    node__model_8_m_m_0_cv2_conv_Conv(tu5->tensor__model_8_m_m_0_cv1_act_Mul_Output_0, tensor_model_8_m_0_cv2_conv_weight, tensor_model_8_m_0_cv2_conv_bias, tu1->tensor__model_8_m_m_0_cv2_conv_Conv_Output_0);
    node__model_8_m_m_0_cv2_act_Sigmoid(tu1->tensor__model_8_m_m_0_cv2_conv_Conv_Output_0, tu4->tensor__model_8_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_8_m_m_0_cv2_act_Mul(tu1->tensor__model_8_m_m_0_cv2_conv_Conv_Output_0, tu4->tensor__model_8_m_m_0_cv2_act_Sigmoid_Output_0, tu5->tensor__model_8_m_m_0_cv2_act_Mul_Output_0);
    node__model_8_m_m_0_Add(tu6->tensor__model_8_cv1_act_Mul_Output_0, tu5->tensor__model_8_m_m_0_cv2_act_Mul_Output_0, tu1->tensor__model_8_m_m_0_Add_Output_0);
    node__model_8_Concat(tu1->tensor__model_8_m_m_0_Add_Output_0, tu0->tensor__model_8_cv2_act_Mul_Output_0, tu4->tensor__model_8_Concat_Output_0);
    node__model_8_cv3_conv_Conv(tu4->tensor__model_8_Concat_Output_0, tensor_model_8_cv3_conv_weight, tensor_model_8_cv3_conv_bias, tu0->tensor__model_8_cv3_conv_Conv_Output_0);
    node__model_8_cv3_act_Sigmoid(tu0->tensor__model_8_cv3_conv_Conv_Output_0, tu1->tensor__model_8_cv3_act_Sigmoid_Output_0);
    node__model_8_cv3_act_Mul(tu0->tensor__model_8_cv3_conv_Conv_Output_0, tu1->tensor__model_8_cv3_act_Sigmoid_Output_0, tu4->tensor__model_8_cv3_act_Mul_Output_0);
    node__model_9_cv1_conv_Conv(tu4->tensor__model_8_cv3_act_Mul_Output_0, tensor_model_9_cv1_conv_weight, tensor_model_9_cv1_conv_bias, tu0->tensor__model_9_cv1_conv_Conv_Output_0);
    node__model_9_cv1_act_Sigmoid(tu0->tensor__model_9_cv1_conv_Conv_Output_0, tu1->tensor__model_9_cv1_act_Sigmoid_Output_0);
    node__model_9_cv1_act_Mul(tu0->tensor__model_9_cv1_conv_Conv_Output_0, tu1->tensor__model_9_cv1_act_Sigmoid_Output_0, tu4->tensor__model_9_cv1_act_Mul_Output_0);
    node__model_9_m_MaxPool(tu4->tensor__model_9_cv1_act_Mul_Output_0, tu0->tensor__model_9_m_MaxPool_Output_0);
    node__model_9_m_1_MaxPool(tu0->tensor__model_9_m_MaxPool_Output_0, tu1->tensor__model_9_m_1_MaxPool_Output_0);
    node__model_9_m_2_MaxPool(tu1->tensor__model_9_m_1_MaxPool_Output_0, tu5->tensor__model_9_m_2_MaxPool_Output_0);
    node__model_9_Concat(tu4->tensor__model_9_cv1_act_Mul_Output_0, tu0->tensor__model_9_m_MaxPool_Output_0, tu1->tensor__model_9_m_1_MaxPool_Output_0, tu5->tensor__model_9_m_2_MaxPool_Output_0, tu6->tensor__model_9_Concat_Output_0);
    node__model_9_cv2_conv_Conv(tu6->tensor__model_9_Concat_Output_0, tensor_model_9_cv2_conv_weight, tensor_model_9_cv2_conv_bias, tu0->tensor__model_9_cv2_conv_Conv_Output_0);
    node__model_9_cv2_act_Sigmoid(tu0->tensor__model_9_cv2_conv_Conv_Output_0, tu1->tensor__model_9_cv2_act_Sigmoid_Output_0);
    node__model_9_cv2_act_Mul(tu0->tensor__model_9_cv2_conv_Conv_Output_0, tu1->tensor__model_9_cv2_act_Sigmoid_Output_0, tu4->tensor__model_9_cv2_act_Mul_Output_0);
    node__model_10_conv_Conv(tu4->tensor__model_9_cv2_act_Mul_Output_0, tensor_model_10_conv_weight, tensor_model_10_conv_bias, tu0->tensor__model_10_conv_Conv_Output_0);
    node__model_10_act_Sigmoid(tu0->tensor__model_10_conv_Conv_Output_0, tu1->tensor__model_10_act_Sigmoid_Output_0);
    node__model_10_act_Mul(tu0->tensor__model_10_conv_Conv_Output_0, tu1->tensor__model_10_act_Sigmoid_Output_0, tu4->tensor__model_10_act_Mul_Output_0);
    node__model_11_Resize(tu4->tensor__model_10_act_Mul_Output_0, tensor__model_11_Concat_1_output_0, tu0->tensor__model_11_Resize_Output_0);
    node__model_12_Concat(tu0->tensor__model_11_Resize_Output_0, tu3->tensor__model_6_cv3_act_Mul_Output_0, tu1->tensor__model_12_Concat_Output_0);
    node__model_13_cv1_conv_Conv(tu1->tensor__model_12_Concat_Output_0, tensor_model_13_cv1_conv_weight, tensor_model_13_cv1_conv_bias, tu0->tensor__model_13_cv1_conv_Conv_Output_0);
    node__model_13_cv2_conv_Conv(tu1->tensor__model_12_Concat_Output_0, tensor_model_13_cv2_conv_weight, tensor_model_13_cv2_conv_bias, tu3->tensor__model_13_cv2_conv_Conv_Output_0);
    node__model_13_cv1_act_Sigmoid(tu0->tensor__model_13_cv1_conv_Conv_Output_0, tu1->tensor__model_13_cv1_act_Sigmoid_Output_0);
    node__model_13_cv2_act_Sigmoid(tu3->tensor__model_13_cv2_conv_Conv_Output_0, tu5->tensor__model_13_cv2_act_Sigmoid_Output_0);
    node__model_13_cv1_act_Mul(tu0->tensor__model_13_cv1_conv_Conv_Output_0, tu1->tensor__model_13_cv1_act_Sigmoid_Output_0, tu6->tensor__model_13_cv1_act_Mul_Output_0);
    node__model_13_cv2_act_Mul(tu3->tensor__model_13_cv2_conv_Conv_Output_0, tu5->tensor__model_13_cv2_act_Sigmoid_Output_0, tu0->tensor__model_13_cv2_act_Mul_Output_0);
    node__model_13_m_m_0_cv1_conv_Conv(tu6->tensor__model_13_cv1_act_Mul_Output_0, tensor_model_13_m_0_cv1_conv_weight, tensor_model_13_m_0_cv1_conv_bias, tu1->tensor__model_13_m_m_0_cv1_conv_Conv_Output_0);
    node__model_13_m_m_0_cv1_act_Sigmoid(tu1->tensor__model_13_m_m_0_cv1_conv_Conv_Output_0, tu3->tensor__model_13_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_13_m_m_0_cv1_act_Mul(tu1->tensor__model_13_m_m_0_cv1_conv_Conv_Output_0, tu3->tensor__model_13_m_m_0_cv1_act_Sigmoid_Output_0, tu5->tensor__model_13_m_m_0_cv1_act_Mul_Output_0);
    node__model_13_m_m_0_cv2_conv_Conv(tu5->tensor__model_13_m_m_0_cv1_act_Mul_Output_0, tensor_model_13_m_0_cv2_conv_weight, tensor_model_13_m_0_cv2_conv_bias, tu1->tensor__model_13_m_m_0_cv2_conv_Conv_Output_0);
    node__model_13_m_m_0_cv2_act_Sigmoid(tu1->tensor__model_13_m_m_0_cv2_conv_Conv_Output_0, tu3->tensor__model_13_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_13_m_m_0_cv2_act_Mul(tu1->tensor__model_13_m_m_0_cv2_conv_Conv_Output_0, tu3->tensor__model_13_m_m_0_cv2_act_Sigmoid_Output_0, tu5->tensor__model_13_m_m_0_cv2_act_Mul_Output_0);
    node__model_13_Concat(tu5->tensor__model_13_m_m_0_cv2_act_Mul_Output_0, tu0->tensor__model_13_cv2_act_Mul_Output_0, tu1->tensor__model_13_Concat_Output_0);
    node__model_13_cv3_conv_Conv(tu1->tensor__model_13_Concat_Output_0, tensor_model_13_cv3_conv_weight, tensor_model_13_cv3_conv_bias, tu0->tensor__model_13_cv3_conv_Conv_Output_0);
    node__model_13_cv3_act_Sigmoid(tu0->tensor__model_13_cv3_conv_Conv_Output_0, tu1->tensor__model_13_cv3_act_Sigmoid_Output_0);
    node__model_13_cv3_act_Mul(tu0->tensor__model_13_cv3_conv_Conv_Output_0, tu1->tensor__model_13_cv3_act_Sigmoid_Output_0, tu3->tensor__model_13_cv3_act_Mul_Output_0);
    node__model_14_conv_Conv(tu3->tensor__model_13_cv3_act_Mul_Output_0, tensor_model_14_conv_weight, tensor_model_14_conv_bias, tu0->tensor__model_14_conv_Conv_Output_0);
    node__model_14_act_Sigmoid(tu0->tensor__model_14_conv_Conv_Output_0, tu1->tensor__model_14_act_Sigmoid_Output_0);
    node__model_14_act_Mul(tu0->tensor__model_14_conv_Conv_Output_0, tu1->tensor__model_14_act_Sigmoid_Output_0, tu3->tensor__model_14_act_Mul_Output_0);
    node__model_15_Resize(tu3->tensor__model_14_act_Mul_Output_0, tensor__model_15_Concat_1_output_0, tu0->tensor__model_15_Resize_Output_0);
    node__model_16_Concat(tu0->tensor__model_15_Resize_Output_0, tu2->tensor__model_4_cv3_act_Mul_Output_0, tu1->tensor__model_16_Concat_Output_0);
    node__model_17_cv1_conv_Conv(tu1->tensor__model_16_Concat_Output_0, tensor_model_17_cv1_conv_weight, tensor_model_17_cv1_conv_bias, tu0->tensor__model_17_cv1_conv_Conv_Output_0);
    node__model_17_cv2_conv_Conv(tu1->tensor__model_16_Concat_Output_0, tensor_model_17_cv2_conv_weight, tensor_model_17_cv2_conv_bias, tu2->tensor__model_17_cv2_conv_Conv_Output_0);
    node__model_17_cv1_act_Sigmoid(tu0->tensor__model_17_cv1_conv_Conv_Output_0, tu1->tensor__model_17_cv1_act_Sigmoid_Output_0);
    node__model_17_cv2_act_Sigmoid(tu2->tensor__model_17_cv2_conv_Conv_Output_0, tu5->tensor__model_17_cv2_act_Sigmoid_Output_0);
    node__model_17_cv1_act_Mul(tu0->tensor__model_17_cv1_conv_Conv_Output_0, tu1->tensor__model_17_cv1_act_Sigmoid_Output_0, tu6->tensor__model_17_cv1_act_Mul_Output_0);
    node__model_17_cv2_act_Mul(tu2->tensor__model_17_cv2_conv_Conv_Output_0, tu5->tensor__model_17_cv2_act_Sigmoid_Output_0, tu0->tensor__model_17_cv2_act_Mul_Output_0);
    node__model_17_m_m_0_cv1_conv_Conv(tu6->tensor__model_17_cv1_act_Mul_Output_0, tensor_model_17_m_0_cv1_conv_weight, tensor_model_17_m_0_cv1_conv_bias, tu1->tensor__model_17_m_m_0_cv1_conv_Conv_Output_0);
    node__model_17_m_m_0_cv1_act_Sigmoid(tu1->tensor__model_17_m_m_0_cv1_conv_Conv_Output_0, tu2->tensor__model_17_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_17_m_m_0_cv1_act_Mul(tu1->tensor__model_17_m_m_0_cv1_conv_Conv_Output_0, tu2->tensor__model_17_m_m_0_cv1_act_Sigmoid_Output_0, tu5->tensor__model_17_m_m_0_cv1_act_Mul_Output_0);
    node__model_17_m_m_0_cv2_conv_Conv(tu5->tensor__model_17_m_m_0_cv1_act_Mul_Output_0, tensor_model_17_m_0_cv2_conv_weight, tensor_model_17_m_0_cv2_conv_bias, tu1->tensor__model_17_m_m_0_cv2_conv_Conv_Output_0);
    node__model_17_m_m_0_cv2_act_Sigmoid(tu1->tensor__model_17_m_m_0_cv2_conv_Conv_Output_0, tu2->tensor__model_17_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_17_m_m_0_cv2_act_Mul(tu1->tensor__model_17_m_m_0_cv2_conv_Conv_Output_0, tu2->tensor__model_17_m_m_0_cv2_act_Sigmoid_Output_0, tu5->tensor__model_17_m_m_0_cv2_act_Mul_Output_0);
    node__model_17_Concat(tu5->tensor__model_17_m_m_0_cv2_act_Mul_Output_0, tu0->tensor__model_17_cv2_act_Mul_Output_0, tu1->tensor__model_17_Concat_Output_0);
    node__model_17_cv3_conv_Conv(tu1->tensor__model_17_Concat_Output_0, tensor_model_17_cv3_conv_weight, tensor_model_17_cv3_conv_bias, tu0->tensor__model_17_cv3_conv_Conv_Output_0);
    node__model_17_cv3_act_Sigmoid(tu0->tensor__model_17_cv3_conv_Conv_Output_0, tu1->tensor__model_17_cv3_act_Sigmoid_Output_0);
    node__model_17_cv3_act_Mul(tu0->tensor__model_17_cv3_conv_Conv_Output_0, tu1->tensor__model_17_cv3_act_Sigmoid_Output_0, tu2->tensor__model_17_cv3_act_Mul_Output_0);
    node__model_18_conv_Conv(tu2->tensor__model_17_cv3_act_Mul_Output_0, tensor_model_18_conv_weight, tensor_model_18_conv_bias, tu0->tensor__model_18_conv_Conv_Output_0);
    node__model_24_m_0_Conv(tu2->tensor__model_17_cv3_act_Mul_Output_0, tensor_model_24_m_0_weight, tensor_model_24_m_0_bias, tu1->tensor__model_24_m_0_Conv_Output_0);
    node__model_18_act_Sigmoid(tu0->tensor__model_18_conv_Conv_Output_0, tu2->tensor__model_18_act_Sigmoid_Output_0);
    node__model_24_Reshape(tu1->tensor__model_24_m_0_Conv_Output_0, tensor__model_24_Constant_output_0, tu5->tensor__model_24_Reshape_Output_0);
    node__model_18_act_Mul(tu0->tensor__model_18_conv_Conv_Output_0, tu2->tensor__model_18_act_Sigmoid_Output_0, tu1->tensor__model_18_act_Mul_Output_0);
    node__model_24_Transpose(tu5->tensor__model_24_Reshape_Output_0, tu0->tensor__model_24_Transpose_Output_0);
    node__model_19_Concat(tu1->tensor__model_18_act_Mul_Output_0, tu3->tensor__model_14_act_Mul_Output_0, tu2->tensor__model_19_Concat_Output_0);
    node__model_24_Sigmoid(tu0->tensor__model_24_Transpose_Output_0, tu1->tensor__model_24_Sigmoid_Output_0);
    node__model_20_cv1_conv_Conv(tu2->tensor__model_19_Concat_Output_0, tensor_model_20_cv1_conv_weight, tensor_model_20_cv1_conv_bias, tu0->tensor__model_20_cv1_conv_Conv_Output_0);
    node__model_20_cv2_conv_Conv(tu2->tensor__model_19_Concat_Output_0, tensor_model_20_cv2_conv_weight, tensor_model_20_cv2_conv_bias, tu3->tensor__model_20_cv2_conv_Conv_Output_0);
    node__model_24_Split(tu1->tensor__model_24_Sigmoid_Output_0, tensor_onnx__Split_377, tu2->tensor__model_24_Split_Output_0, tu5->tensor__model_24_Split_Output_1, tu6->tensor__model_24_Split_Output_2);
    node__model_20_cv1_act_Sigmoid(tu0->tensor__model_20_cv1_conv_Conv_Output_0, tu1->tensor__model_20_cv1_act_Sigmoid_Output_0);
    node__model_20_cv2_act_Sigmoid(tu3->tensor__model_20_cv2_conv_Conv_Output_0, tu7->tensor__model_20_cv2_act_Sigmoid_Output_0);
    node__model_24_Mul(tu2->tensor__model_24_Split_Output_0, tensor__model_24_Constant_1_output_0, tu8->tensor__model_24_Mul_Output_0);
    node__model_24_Mul_2(tu5->tensor__model_24_Split_Output_1, tensor__model_24_Constant_1_output_0, tu2->tensor__model_24_Mul_2_Output_0);
    node__model_20_cv1_act_Mul(tu0->tensor__model_20_cv1_conv_Conv_Output_0, tu1->tensor__model_20_cv1_act_Sigmoid_Output_0, tu5->tensor__model_20_cv1_act_Mul_Output_0);
    node__model_20_cv2_act_Mul(tu3->tensor__model_20_cv2_conv_Conv_Output_0, tu7->tensor__model_20_cv2_act_Sigmoid_Output_0, tu0->tensor__model_20_cv2_act_Mul_Output_0);
    node__model_24_Add(tu8->tensor__model_24_Mul_Output_0, tensor__model_24_Constant_2_output_0, tu1->tensor__model_24_Add_Output_0);
    node__model_24_Pow(tu2->tensor__model_24_Mul_2_Output_0, tensor__model_24_Constant_5_output_0, tu3->tensor__model_24_Pow_Output_0);
    node__model_20_m_m_0_cv1_conv_Conv(tu5->tensor__model_20_cv1_act_Mul_Output_0, tensor_model_20_m_0_cv1_conv_weight, tensor_model_20_m_0_cv1_conv_bias, tu2->tensor__model_20_m_m_0_cv1_conv_Conv_Output_0);
    node__model_24_Mul_1(tu1->tensor__model_24_Add_Output_0, tensor__model_24_Constant_3_output_0, tu5->tensor__model_24_Mul_1_Output_0);
    node__model_24_Mul_3(tu3->tensor__model_24_Pow_Output_0, tensor__model_24_Constant_6_output_0, tu1->tensor__model_24_Mul_3_Output_0);
    node__model_20_m_m_0_cv1_act_Sigmoid(tu2->tensor__model_20_m_m_0_cv1_conv_Conv_Output_0, tu3->tensor__model_20_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_24_Concat(tu5->tensor__model_24_Mul_1_Output_0, tu1->tensor__model_24_Mul_3_Output_0, tu6->tensor__model_24_Split_Output_2, tu7->tensor__model_24_Concat_Output_0);
    node__model_20_m_m_0_cv1_act_Mul(tu2->tensor__model_20_m_m_0_cv1_conv_Conv_Output_0, tu3->tensor__model_20_m_m_0_cv1_act_Sigmoid_Output_0, tu1->tensor__model_20_m_m_0_cv1_act_Mul_Output_0);
    node__model_24_Reshape_1(tu7->tensor__model_24_Concat_Output_0, tensor__model_24_Constant_7_output_0, tu2->tensor__model_24_Reshape_1_Output_0);
    node__model_20_m_m_0_cv2_conv_Conv(tu1->tensor__model_20_m_m_0_cv1_act_Mul_Output_0, tensor_model_20_m_0_cv2_conv_weight, tensor_model_20_m_0_cv2_conv_bias, tu3->tensor__model_20_m_m_0_cv2_conv_Conv_Output_0);
    node__model_20_m_m_0_cv2_act_Sigmoid(tu3->tensor__model_20_m_m_0_cv2_conv_Conv_Output_0, tu1->tensor__model_20_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_20_m_m_0_cv2_act_Mul(tu3->tensor__model_20_m_m_0_cv2_conv_Conv_Output_0, tu1->tensor__model_20_m_m_0_cv2_act_Sigmoid_Output_0, tu5->tensor__model_20_m_m_0_cv2_act_Mul_Output_0);
    node__model_20_Concat(tu5->tensor__model_20_m_m_0_cv2_act_Mul_Output_0, tu0->tensor__model_20_cv2_act_Mul_Output_0, tu1->tensor__model_20_Concat_Output_0);
    node__model_20_cv3_conv_Conv(tu1->tensor__model_20_Concat_Output_0, tensor_model_20_cv3_conv_weight, tensor_model_20_cv3_conv_bias, tu0->tensor__model_20_cv3_conv_Conv_Output_0);
    node__model_20_cv3_act_Sigmoid(tu0->tensor__model_20_cv3_conv_Conv_Output_0, tu1->tensor__model_20_cv3_act_Sigmoid_Output_0);
    node__model_20_cv3_act_Mul(tu0->tensor__model_20_cv3_conv_Conv_Output_0, tu1->tensor__model_20_cv3_act_Sigmoid_Output_0, tu3->tensor__model_20_cv3_act_Mul_Output_0);
    node__model_21_conv_Conv(tu3->tensor__model_20_cv3_act_Mul_Output_0, tensor_model_21_conv_weight, tensor_model_21_conv_bias, tu0->tensor__model_21_conv_Conv_Output_0);
    node__model_24_m_1_Conv(tu3->tensor__model_20_cv3_act_Mul_Output_0, tensor_model_24_m_1_weight, tensor_model_24_m_1_bias, tu1->tensor__model_24_m_1_Conv_Output_0);
    node__model_21_act_Sigmoid(tu0->tensor__model_21_conv_Conv_Output_0, tu3->tensor__model_21_act_Sigmoid_Output_0);
    node__model_24_Reshape_2(tu1->tensor__model_24_m_1_Conv_Output_0, tensor__model_24_Constant_8_output_0, tu5->tensor__model_24_Reshape_2_Output_0);
    node__model_21_act_Mul(tu0->tensor__model_21_conv_Conv_Output_0, tu3->tensor__model_21_act_Sigmoid_Output_0, tu1->tensor__model_21_act_Mul_Output_0);
    node__model_24_Transpose_1(tu5->tensor__model_24_Reshape_2_Output_0, tu0->tensor__model_24_Transpose_1_Output_0);
    node__model_22_Concat(tu1->tensor__model_21_act_Mul_Output_0, tu4->tensor__model_10_act_Mul_Output_0, tu3->tensor__model_22_Concat_Output_0);
    node__model_24_Sigmoid_1(tu0->tensor__model_24_Transpose_1_Output_0, tu1->tensor__model_24_Sigmoid_1_Output_0);
    node__model_23_cv1_conv_Conv(tu3->tensor__model_22_Concat_Output_0, tensor_model_23_cv1_conv_weight, tensor_model_23_cv1_conv_bias, tu0->tensor__model_23_cv1_conv_Conv_Output_0);
    node__model_23_cv2_conv_Conv(tu3->tensor__model_22_Concat_Output_0, tensor_model_23_cv2_conv_weight, tensor_model_23_cv2_conv_bias, tu4->tensor__model_23_cv2_conv_Conv_Output_0);
    node__model_24_Split_1(tu1->tensor__model_24_Sigmoid_1_Output_0, tensor_onnx__Split_377, tu3->tensor__model_24_Split_1_Output_0, tu5->tensor__model_24_Split_1_Output_1, tu6->tensor__model_24_Split_1_Output_2);
    node__model_23_cv1_act_Sigmoid(tu0->tensor__model_23_cv1_conv_Conv_Output_0, tu1->tensor__model_23_cv1_act_Sigmoid_Output_0);
    node__model_23_cv2_act_Sigmoid(tu4->tensor__model_23_cv2_conv_Conv_Output_0, tu7->tensor__model_23_cv2_act_Sigmoid_Output_0);
    node__model_24_Mul_4(tu3->tensor__model_24_Split_1_Output_0, tensor__model_24_Constant_1_output_0, tu8->tensor__model_24_Mul_4_Output_0);
    node__model_24_Mul_6(tu5->tensor__model_24_Split_1_Output_1, tensor__model_24_Constant_1_output_0, tu3->tensor__model_24_Mul_6_Output_0);
    node__model_23_cv1_act_Mul(tu0->tensor__model_23_cv1_conv_Conv_Output_0, tu1->tensor__model_23_cv1_act_Sigmoid_Output_0, tu5->tensor__model_23_cv1_act_Mul_Output_0);
    node__model_23_cv2_act_Mul(tu4->tensor__model_23_cv2_conv_Conv_Output_0, tu7->tensor__model_23_cv2_act_Sigmoid_Output_0, tu0->tensor__model_23_cv2_act_Mul_Output_0);
    node__model_24_Add_1(tu8->tensor__model_24_Mul_4_Output_0, tensor__model_24_Constant_10_output_0, tu1->tensor__model_24_Add_1_Output_0);
    node__model_24_Pow_1(tu3->tensor__model_24_Mul_6_Output_0, tensor__model_24_Constant_5_output_0, tu4->tensor__model_24_Pow_1_Output_0);
    node__model_23_m_m_0_cv1_conv_Conv(tu5->tensor__model_23_cv1_act_Mul_Output_0, tensor_model_23_m_0_cv1_conv_weight, tensor_model_23_m_0_cv1_conv_bias, tu3->tensor__model_23_m_m_0_cv1_conv_Conv_Output_0);
    node__model_24_Mul_5(tu1->tensor__model_24_Add_1_Output_0, tensor__model_24_Constant_11_output_0, tu5->tensor__model_24_Mul_5_Output_0);
    node__model_24_Mul_7(tu4->tensor__model_24_Pow_1_Output_0, tensor__model_24_Constant_14_output_0, tu1->tensor__model_24_Mul_7_Output_0);
    node__model_23_m_m_0_cv1_act_Sigmoid(tu3->tensor__model_23_m_m_0_cv1_conv_Conv_Output_0, tu4->tensor__model_23_m_m_0_cv1_act_Sigmoid_Output_0);
    node__model_24_Concat_1(tu5->tensor__model_24_Mul_5_Output_0, tu1->tensor__model_24_Mul_7_Output_0, tu6->tensor__model_24_Split_1_Output_2, tu7->tensor__model_24_Concat_1_Output_0);
    node__model_23_m_m_0_cv1_act_Mul(tu3->tensor__model_23_m_m_0_cv1_conv_Conv_Output_0, tu4->tensor__model_23_m_m_0_cv1_act_Sigmoid_Output_0, tu1->tensor__model_23_m_m_0_cv1_act_Mul_Output_0);
    node__model_24_Reshape_3(tu7->tensor__model_24_Concat_1_Output_0, tensor__model_24_Constant_15_output_0, tu3->tensor__model_24_Reshape_3_Output_0);
    node__model_23_m_m_0_cv2_conv_Conv(tu1->tensor__model_23_m_m_0_cv1_act_Mul_Output_0, tensor_model_23_m_0_cv2_conv_weight, tensor_model_23_m_0_cv2_conv_bias, tu4->tensor__model_23_m_m_0_cv2_conv_Conv_Output_0);
    node__model_23_m_m_0_cv2_act_Sigmoid(tu4->tensor__model_23_m_m_0_cv2_conv_Conv_Output_0, tu1->tensor__model_23_m_m_0_cv2_act_Sigmoid_Output_0);
    node__model_23_m_m_0_cv2_act_Mul(tu4->tensor__model_23_m_m_0_cv2_conv_Conv_Output_0, tu1->tensor__model_23_m_m_0_cv2_act_Sigmoid_Output_0, tu5->tensor__model_23_m_m_0_cv2_act_Mul_Output_0);
    node__model_23_Concat(tu5->tensor__model_23_m_m_0_cv2_act_Mul_Output_0, tu0->tensor__model_23_cv2_act_Mul_Output_0, tu1->tensor__model_23_Concat_Output_0);
    node__model_23_cv3_conv_Conv(tu1->tensor__model_23_Concat_Output_0, tensor_model_23_cv3_conv_weight, tensor_model_23_cv3_conv_bias, tu0->tensor__model_23_cv3_conv_Conv_Output_0);
    node__model_23_cv3_act_Sigmoid(tu0->tensor__model_23_cv3_conv_Conv_Output_0, tu1->tensor__model_23_cv3_act_Sigmoid_Output_0);
    node__model_23_cv3_act_Mul(tu0->tensor__model_23_cv3_conv_Conv_Output_0, tu1->tensor__model_23_cv3_act_Sigmoid_Output_0, tu4->tensor__model_23_cv3_act_Mul_Output_0);
    node__model_24_m_2_Conv(tu4->tensor__model_23_cv3_act_Mul_Output_0, tensor_model_24_m_2_weight, tensor_model_24_m_2_bias, tu0->tensor__model_24_m_2_Conv_Output_0);
    node__model_24_Reshape_4(tu0->tensor__model_24_m_2_Conv_Output_0, tensor__model_24_Constant_16_output_0, tu1->tensor__model_24_Reshape_4_Output_0);
    node__model_24_Transpose_2(tu1->tensor__model_24_Reshape_4_Output_0, tu0->tensor__model_24_Transpose_2_Output_0);
    node__model_24_Sigmoid_2(tu0->tensor__model_24_Transpose_2_Output_0, tu1->tensor__model_24_Sigmoid_2_Output_0);
    node__model_24_Split_2(tu1->tensor__model_24_Sigmoid_2_Output_0, tensor_onnx__Split_377, tu0->tensor__model_24_Split_2_Output_0, tu4->tensor__model_24_Split_2_Output_1, tu5->tensor__model_24_Split_2_Output_2);
    node__model_24_Mul_8(tu0->tensor__model_24_Split_2_Output_0, tensor__model_24_Constant_1_output_0, tu1->tensor__model_24_Mul_8_Output_0);
    node__model_24_Mul_10(tu4->tensor__model_24_Split_2_Output_1, tensor__model_24_Constant_1_output_0, tu0->tensor__model_24_Mul_10_Output_0);
    node__model_24_Add_2(tu1->tensor__model_24_Mul_8_Output_0, tensor__model_24_Constant_18_output_0, tu4->tensor__model_24_Add_2_Output_0);
    node__model_24_Pow_2(tu0->tensor__model_24_Mul_10_Output_0, tensor__model_24_Constant_5_output_0, tu1->tensor__model_24_Pow_2_Output_0);
    node__model_24_Mul_9(tu4->tensor__model_24_Add_2_Output_0, tensor__model_24_Constant_19_output_0, tu0->tensor__model_24_Mul_9_Output_0);
    node__model_24_Mul_11(tu1->tensor__model_24_Pow_2_Output_0, tensor__model_24_Constant_22_output_0, tu4->tensor__model_24_Mul_11_Output_0);
    node__model_24_Concat_2(tu0->tensor__model_24_Mul_9_Output_0, tu4->tensor__model_24_Mul_11_Output_0, tu5->tensor__model_24_Split_2_Output_2, tu1->tensor__model_24_Concat_2_Output_0);
    node__model_24_Reshape_5(tu1->tensor__model_24_Concat_2_Output_0, tensor__model_24_Constant_23_output_0, tu0->tensor__model_24_Reshape_5_Output_0);
    node__model_24_Concat_3(tu2->tensor__model_24_Reshape_1_Output_0, tu3->tensor__model_24_Reshape_3_Output_0, tu0->tensor__model_24_Reshape_5_Output_0, output0);
    
    free(tu0);
    free(tu1);
    free(tu2);
    free(tu3);
    free(tu4);
    free(tu5);
    free(tu6);
    free(tu7);
    free(tu8);
}
