#ifndef YOLO5N_H
#define YOLO5N_H
#include <stdint.h>
#include <stdbool.h>
union tensor_union_0
{
    float tensor__model_0_conv_Conv_Output_0[1][16][112][112]; // 200704
    float tensor__model_1_conv_Conv_Output_0[1][32][56][56]; // 100352
    float tensor__model_2_cv1_conv_Conv_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_cv2_act_Mul_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_cv3_conv_Conv_Output_0[1][32][56][56]; // 100352
    float tensor__model_3_conv_Conv_Output_0[1][64][28][28]; // 50176
    float tensor__model_4_cv1_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_cv2_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_cv3_conv_Conv_Output_0[1][64][28][28]; // 50176
    float tensor__model_5_conv_Conv_Output_0[1][128][14][14]; // 25088
    float tensor__model_6_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_cv3_conv_Conv_Output_0[1][128][14][14]; // 25088
    float tensor__model_7_conv_Conv_Output_0[1][256][7][7]; // 12544
    float tensor__model_8_cv1_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_cv2_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_cv3_conv_Conv_Output_0[1][256][7][7]; // 12544
    float tensor__model_9_cv1_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_9_m_MaxPool_Output_0[1][128][7][7]; // 6272
    float tensor__model_9_cv2_conv_Conv_Output_0[1][256][7][7]; // 12544
    float tensor__model_10_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_11_Resize_Output_0[1][128][14][14]; // 25088
    float tensor__model_13_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_cv3_conv_Conv_Output_0[1][128][14][14]; // 25088
    float tensor__model_14_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_15_Resize_Output_0[1][64][28][28]; // 50176
    float tensor__model_17_cv1_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_cv2_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_cv3_conv_Conv_Output_0[1][64][28][28]; // 50176
    float tensor__model_18_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Transpose_Output_0[1][3][28][28][85]; // 199920
    float tensor__model_20_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_20_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_20_cv3_conv_Conv_Output_0[1][128][14][14]; // 25088
    float tensor__model_21_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Transpose_1_Output_0[1][3][14][14][85]; // 49980
    float tensor__model_23_cv1_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_23_cv2_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_23_cv3_conv_Conv_Output_0[1][256][7][7]; // 12544
    float tensor__model_24_m_2_Conv_Output_0[1][255][7][7]; // 12495
    float tensor__model_24_Transpose_2_Output_0[1][3][7][7][85]; // 12495
    float tensor__model_24_Split_2_Output_0[1][3][7][7][2]; // 294
    float tensor__model_24_Mul_10_Output_0[1][3][7][7][2]; // 294
    float tensor__model_24_Mul_9_Output_0[1][3][7][7][2]; // 294
    float tensor__model_24_Reshape_5_Output_0[1][147][85]; // 12495
};
//static union tensor_union_0 tu0;
union tensor_union_1
{
    float tensor__model_0_act_Sigmoid_Output_0[1][16][112][112]; // 200704
    float tensor__model_1_act_Sigmoid_Output_0[1][32][56][56]; // 100352
    float tensor__model_2_cv2_conv_Conv_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_m_m_0_cv1_conv_Conv_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_m_m_0_cv2_conv_Conv_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_m_m_0_Add_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_cv3_act_Sigmoid_Output_0[1][32][56][56]; // 100352
    float tensor__model_3_act_Sigmoid_Output_0[1][64][28][28]; // 50176
    float tensor__model_4_cv2_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_0_cv1_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_0_cv2_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_0_Add_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_Concat_Output_0[1][64][28][28]; // 50176
    float tensor__model_4_cv3_act_Sigmoid_Output_0[1][64][28][28]; // 50176
    float tensor__model_5_act_Sigmoid_Output_0[1][128][14][14]; // 25088
    float tensor__model_6_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_0_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_0_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_0_Add_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_2_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_2_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_2_Add_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_cv3_act_Sigmoid_Output_0[1][128][14][14]; // 25088
    float tensor__model_7_act_Sigmoid_Output_0[1][256][7][7]; // 12544
    float tensor__model_8_cv2_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_m_m_0_cv1_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_m_m_0_cv2_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_m_m_0_Add_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_cv3_act_Sigmoid_Output_0[1][256][7][7]; // 12544
    float tensor__model_9_cv1_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_9_m_1_MaxPool_Output_0[1][128][7][7]; // 6272
    float tensor__model_9_cv2_act_Sigmoid_Output_0[1][256][7][7]; // 12544
    float tensor__model_10_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_12_Concat_Output_0[1][256][14][14]; // 50176
    float tensor__model_13_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_m_m_0_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_m_m_0_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_Concat_Output_0[1][128][14][14]; // 25088
    float tensor__model_13_cv3_act_Sigmoid_Output_0[1][128][14][14]; // 25088
    float tensor__model_14_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_16_Concat_Output_0[1][128][28][28]; // 100352
    float tensor__model_17_cv1_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_m_m_0_cv1_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_m_m_0_cv2_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_Concat_Output_0[1][64][28][28]; // 50176
    float tensor__model_17_cv3_act_Sigmoid_Output_0[1][64][28][28]; // 50176
    float tensor__model_24_m_0_Conv_Output_0[1][255][28][28]; // 199920
    float tensor__model_18_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Sigmoid_Output_0[1][3][28][28][85]; // 199920
    float tensor__model_20_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Add_Output_0[1][3][28][28][2]; // 4704
    float tensor__model_24_Mul_3_Output_0[1][3][28][28][2]; // 4704
    float tensor__model_20_m_m_0_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_20_m_m_0_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_20_Concat_Output_0[1][128][14][14]; // 25088
    float tensor__model_20_cv3_act_Sigmoid_Output_0[1][128][14][14]; // 25088
    float tensor__model_24_m_1_Conv_Output_0[1][255][14][14]; // 49980
    float tensor__model_21_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Sigmoid_1_Output_0[1][3][14][14][85]; // 49980
    float tensor__model_23_cv1_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Add_1_Output_0[1][3][14][14][2]; // 1176
    float tensor__model_24_Mul_7_Output_0[1][3][14][14][2]; // 1176
    float tensor__model_23_m_m_0_cv1_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_23_m_m_0_cv2_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_23_Concat_Output_0[1][256][7][7]; // 12544
    float tensor__model_23_cv3_act_Sigmoid_Output_0[1][256][7][7]; // 12544
    float tensor__model_24_Reshape_4_Output_0[1][3][85][7][7]; // 12495
    float tensor__model_24_Sigmoid_2_Output_0[1][3][7][7][85]; // 12495
    float tensor__model_24_Mul_8_Output_0[1][3][7][7][2]; // 294
    float tensor__model_24_Pow_2_Output_0[1][3][7][7][2]; // 294
    float tensor__model_24_Concat_2_Output_0[1][3][7][7][85]; // 12495
};
//static union tensor_union_1 tu1;
union tensor_union_2
{
    float tensor__model_0_act_Mul_Output_0[1][16][112][112]; // 200704
    float tensor__model_1_act_Mul_Output_0[1][32][56][56]; // 100352
    float tensor__model_2_cv1_act_Sigmoid_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_m_m_0_cv1_act_Sigmoid_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_m_m_0_cv2_act_Sigmoid_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_Concat_Output_0[1][32][56][56]; // 100352
    float tensor__model_2_cv3_act_Mul_Output_0[1][32][56][56]; // 100352
    float tensor__model_3_act_Mul_Output_0[1][64][28][28]; // 50176
    float tensor__model_4_cv1_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_0_cv1_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_0_cv2_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_1_cv1_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_1_cv2_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_1_Add_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_cv3_act_Mul_Output_0[1][64][28][28]; // 50176
    float tensor__model_17_cv2_conv_Conv_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_m_m_0_cv1_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_m_m_0_cv2_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_cv3_act_Mul_Output_0[1][64][28][28]; // 50176
    float tensor__model_18_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_19_Concat_Output_0[1][128][14][14]; // 25088
    float tensor__model_24_Split_Output_0[1][3][28][28][2]; // 4704
    float tensor__model_24_Mul_2_Output_0[1][3][28][28][2]; // 4704
    float tensor__model_20_m_m_0_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Reshape_1_Output_0[1][2352][85]; // 199920
};
//static union tensor_union_2 tu2;
union tensor_union_3
{
    float tensor__model_2_cv2_act_Sigmoid_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_m_m_0_cv1_act_Mul_Output_0[1][16][56][56]; // 50176
    float tensor__model_2_m_m_0_cv2_act_Mul_Output_0[1][16][56][56]; // 50176
    float tensor__model_4_cv2_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_0_cv1_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_0_cv2_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_1_cv1_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_1_cv2_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_5_act_Mul_Output_0[1][128][14][14]; // 25088
    float tensor__model_6_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_0_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_0_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_1_cv1_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_1_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_1_Add_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_Concat_Output_0[1][128][14][14]; // 25088
    float tensor__model_6_cv3_act_Mul_Output_0[1][128][14][14]; // 25088
    float tensor__model_13_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_m_m_0_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_m_m_0_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_cv3_act_Mul_Output_0[1][128][14][14]; // 25088
    float tensor__model_14_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_20_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Pow_Output_0[1][3][28][28][2]; // 4704
    float tensor__model_20_m_m_0_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_20_m_m_0_cv2_conv_Conv_Output_0[1][64][14][14]; // 12544
    float tensor__model_20_cv3_act_Mul_Output_0[1][128][14][14]; // 25088
    float tensor__model_21_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_22_Concat_Output_0[1][256][7][7]; // 12544
    float tensor__model_24_Split_1_Output_0[1][3][14][14][2]; // 1176
    float tensor__model_24_Mul_6_Output_0[1][3][14][14][2]; // 1176
    float tensor__model_23_m_m_0_cv1_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Reshape_3_Output_0[1][588][85]; // 49980
};
//static union tensor_union_3 tu3;
union tensor_union_4
{
    float tensor__model_2_cv1_act_Mul_Output_0[1][16][56][56]; // 50176
    float tensor__model_4_cv1_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_1_cv1_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_4_m_m_1_cv2_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_6_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_0_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_0_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_1_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_1_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_2_cv1_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_2_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_7_act_Mul_Output_0[1][256][7][7]; // 12544
    float tensor__model_8_cv1_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_m_m_0_cv1_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_m_m_0_cv2_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_Concat_Output_0[1][256][7][7]; // 12544
    float tensor__model_8_cv3_act_Mul_Output_0[1][256][7][7]; // 12544
    float tensor__model_9_cv1_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_9_cv2_act_Mul_Output_0[1][256][7][7]; // 12544
    float tensor__model_10_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_23_cv2_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Pow_1_Output_0[1][3][14][14][2]; // 1176
    float tensor__model_23_m_m_0_cv1_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_23_m_m_0_cv2_conv_Conv_Output_0[1][128][7][7]; // 6272
    float tensor__model_23_cv3_act_Mul_Output_0[1][256][7][7]; // 12544
    float tensor__model_24_Split_2_Output_1[1][3][7][7][2]; // 294
    float tensor__model_24_Add_2_Output_0[1][3][7][7][2]; // 294
    float tensor__model_24_Mul_11_Output_0[1][3][7][7][2]; // 294
};
//static union tensor_union_4 tu4;
union tensor_union_5
{
    float tensor__model_6_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_1_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_1_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_2_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_6_m_m_2_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_8_cv2_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_m_m_0_cv1_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_8_m_m_0_cv2_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_9_m_2_MaxPool_Output_0[1][128][7][7]; // 6272
    float tensor__model_13_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_m_m_0_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_13_m_m_0_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_17_cv2_act_Sigmoid_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_m_m_0_cv1_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_17_m_m_0_cv2_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_24_Reshape_Output_0[1][3][85][28][28]; // 199920
    float tensor__model_24_Split_Output_1[1][3][28][28][2]; // 4704
    float tensor__model_20_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Mul_1_Output_0[1][3][28][28][2]; // 4704
    float tensor__model_20_m_m_0_cv2_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Reshape_2_Output_0[1][3][85][14][14]; // 49980
    float tensor__model_24_Split_1_Output_1[1][3][14][14][2]; // 1176
    float tensor__model_23_cv1_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Mul_5_Output_0[1][3][14][14][2]; // 1176
    float tensor__model_23_m_m_0_cv2_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Split_2_Output_2[1][3][7][7][81]; // 11907
};
//static union tensor_union_5 tu5;
union tensor_union_6
{
    float tensor__model_8_cv1_act_Mul_Output_0[1][128][7][7]; // 6272
    float tensor__model_9_Concat_Output_0[1][512][7][7]; // 25088
    float tensor__model_13_cv1_act_Mul_Output_0[1][64][14][14]; // 12544
    float tensor__model_17_cv1_act_Mul_Output_0[1][32][28][28]; // 25088
    float tensor__model_24_Split_Output_2[1][3][28][28][81]; // 190512
    float tensor__model_24_Split_1_Output_2[1][3][14][14][81]; // 47628
};
//static union tensor_union_6 tu6;
union tensor_union_7
{
    float tensor__model_20_cv2_act_Sigmoid_Output_0[1][64][14][14]; // 12544
    float tensor__model_24_Concat_Output_0[1][3][28][28][85]; // 199920
    float tensor__model_23_cv2_act_Sigmoid_Output_0[1][128][7][7]; // 6272
    float tensor__model_24_Concat_1_Output_0[1][3][14][14][85]; // 49980
};
//static union tensor_union_7 tu7;
union tensor_union_8
{
    float tensor__model_24_Mul_Output_0[1][3][28][28][2]; // 4704
    float tensor__model_24_Mul_4_Output_0[1][3][14][14][2]; // 1176
};
//static union tensor_union_8 tu8;
void forward_pass(const float images[1][3][224][224], float output0[1][3087][85]);
#endif // YOLO5N_H