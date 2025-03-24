#include <TFT_eSPI.h>  // Include the TFT library
#include "images.h"
#include "yolo5n.h"

TFT_eSPI tft = TFT_eSPI();  // Create TFT instance

#define I_HEIGHT 250
#define I_WIDTH 250
#define DST_WIDTH 224 // model input
#define DST_HEIGHT 224 // model input

#define IOU_THRESHOLD 0.45f  // Adjust as needed


#define original_width 420   // Target screen width
#define original_height 320  // Target screen height

#define NUM_BOXES 3087
#define NUM_CLASSES 80
#define CONFIDENCE_THRESHOLD 0.5

typedef struct {
  float x, y, w, h;
  float confidence;
  float class_scores;
  int class_id;
} Detection;
// Helper function to compute Intersection over Union (IoU) between two detections
float compute_iou(Detection a, Detection b) {
  int x_left = fmax(a.x, b.x);
  int y_top = fmax(a.y, b.y);
  int x_right = fmin(a.x + a.w, b.x + b.w);
  int y_bottom = fmin(a.y + a.h, b.y + b.h);

  if (x_right < x_left || y_bottom < y_top)
    return 0.0f;

  int intersection_area = (x_right - x_left) * (y_bottom - y_top);
  int area_a = a.w * a.h;
  int area_b = b.w * b.h;
  int union_area = area_a + area_b - intersection_area;

  return (float)intersection_area / union_area;
}

// Non-Maximum Suppression to filter out overlapping detections
void non_maximum_suppression(Detection detections[], int *det_count, float iou_threshold) {
  // Simple O(n^2) NMS based on the combined score
  for (int i = 0; i < *det_count; i++) {
    // Skip suppressed detections (confidence == 0)
    if (detections[i].confidence <= 0)
      continue;
    for (int j = i + 1; j < *det_count; j++) {
      if (detections[j].confidence <= 0)
        continue;
      // If the boxes overlap more than the threshold, suppress the lower score box.
      if (compute_iou(detections[i], detections[j]) > iou_threshold) {
        // Here, we simply suppress detection j.
        // You could also compare scores and choose which to keep.
        detections[j].confidence = 0;
      }
    }
  }

  // Compact the detections array to remove suppressed detections
  int new_count = 0;
  for (int i = 0; i < *det_count; i++) {
    if (detections[i].confidence > 0) {
      detections[new_count++] = detections[i];
    }
  }
  *det_count = new_count;
}
void parse_yolo_output(float output[NUM_BOXES][85], Detection detections[], int *det_count) {
  *det_count = 0;
  float scale_x = (float)original_width / (float)DST_WIDTH;
  float scale_y = (float)original_height / (float)DST_HEIGHT;
  for (int i = 0; i < NUM_BOXES; i++) {
    float confidence = output[i][4];  // Confidence
    if (confidence < CONFIDENCE_THRESHOLD) continue;

    detections[*det_count].x = output[i][0];
    detections[*det_count].y = output[i][1];
    detections[*det_count].w = output[i][2];
    detections[*det_count].h = output[i][3];
    detections[*det_count].confidence = confidence;

    // Get class with highest probability
    // Find class with maximum score
    float max_class_score = -INFINITY;
    int class_id = -1;
    for (int j = 5; j < 85; j++) {
      if (output[i][j] > max_class_score) {
        max_class_score = output[i][j];
        class_id = j - 5;
      }
    }

    float combined_score = max_class_score * confidence;
    if (combined_score < 0.4f) continue;

    // Extract bounding box parameters
    float cx = output[i][0];
    float cy = output[i][1];
    float w = output[i][2];
    float h = output[i][3];
    // Convert to image coordinates
    int x_min = (int)((cx - w / 2.0f) * scale_x);
    int y_min = (int)((cy - h / 2.0f) * scale_y);
    int x_max = (int)((cx + w / 2.0f) * scale_x);
    int y_max = (int)((cy + h / 2.0f) * scale_y);
    // Clamp coordinates to image dimensions
    x_min = fmax(0, fmin(x_min, original_width - 1));
    y_min = fmax(0, fmin(y_min, original_height - 1));
    x_max = fmax(0, fmin(x_max, original_width - 1));
    y_max = fmax(0, fmin(y_max, original_height - 1));

    //detections[*det_count].class_id = best_class;
    detections[*det_count].x = x_min;
    detections[*det_count].y = y_min;
    detections[*det_count].w = x_max - x_min;  // Width
    detections[*det_count].h = y_max - y_min;  // Height
    detections[*det_count].class_scores = combined_score;
    detections[*det_count].class_id = class_id;
    (*det_count)++;
  }
  non_maximum_suppression(detections, det_count, IOU_THRESHOLD);
}


void normalizeImage(uint16_t input[DST_HEIGHT][DST_WIDTH], float output[1][3][DST_HEIGHT][DST_WIDTH]) {
  for (int i = 0; i < DST_WIDTH; ++i) {
    for (int j = 0; j < DST_HEIGHT; ++j) {
      // Extract RGB components from 16-bit RGB565
      uint16_t pixel = input[i][j];
      uint8_t r = (pixel >> 11) & 0x1F;  // Red (5 bits)
      uint8_t g = (pixel >> 5) & 0x3F;   // Green (6 bits)
      uint8_t b = pixel & 0x1F;          // Blue (5 bits)

      // **Convert RGB565 to RGB888**
      uint8_t r8 = (r << 3) | (r >> 2);  // Expand 5-bit red to 8-bit
      uint8_t g8 = (g << 2) | (g >> 4);  // Expand 6-bit green to 8-bit
      uint8_t b8 = (b << 3) | (b >> 2);  // Expand 5-bit blue to 8-bit

      // **Normalize RGB888 to [0, 1] range**
      output[0][0][i][j] = (float)r / 31.0f;
      output[0][1][i][j] = (float)g / 64.0f;
      output[0][2][i][j] = (float)b / 31.0f;
    }
  }
}

// Function to resize image using nearest-neighbor scaling
void resizeImage(const uint16_t *src, uint16_t *dst) {
  for (int y = 0; y < DST_HEIGHT; y++) {
    for (int x = 0; x < DST_WIDTH; x++) {
      int srcX = (x * I_WIDTH) / DST_WIDTH;                 // Scale X
      int srcY = (y * I_HEIGHT) / DST_HEIGHT;               // Scale Y
      dst[y * DST_WIDTH + x] = src[srcY * I_WIDTH + srcX];  // Copy pixel
    }
  }
}

// Function to resize image using nearest-neighbor scaling
void resizeImage_up(const uint16_t *src, uint16_t *dst) {
  for (int y = 0; y < original_height; y++) {
    for (int x = 0; x < original_width; x++) {
      int srcX = (x * I_WIDTH) / original_width;                 // Scale X
      int srcY = (y * I_HEIGHT) / original_height;               // Scale Y
      dst[y * original_width + x] = src[srcY * I_WIDTH + srcX];  // Copy pixel
    }
  }
}

void drawImage(int x_offset, int y_offset, const uint16_t *imageArray, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int index = y * width + x;
      tft.drawPixel(x + x_offset, y + y_offset, imageArray[index]);
    }
  }
}

void setup() {
  Serial.begin(115200);
  tft.init();          // Initialize the display
  tft.setRotation(3);  // Set rotation (0-3)
  tft.setSwapBytes(true);
  // Test display colors
  tft.fillScreen(TFT_WHITE);
  delay(500);
  if (psramInit()) {
    Serial.println("PSRAM is available and initialized.");
    Serial.printf("Free heap: %d\n", esp_get_free_heap_size());
    Serial.printf("Free PSRAM: %d\n", esp_get_free_internal_heap_size());
  } else {
    Serial.println("PSRAM initialization failed!");
  }
  /* Display Image*/
  uint16_t *SrcImage = (uint16_t *)ps_malloc((original_width * original_height) * sizeof(uint16_t));
  resizeImage_up(*picture_1, SrcImage);
  drawImage(0, 0, SrcImage, original_width, original_height);
  free(SrcImage);

  uint16_t *resizedImage = (uint16_t *)ps_malloc((DST_WIDTH * DST_HEIGHT) * sizeof(uint16_t));
  float(*normalized)[3][DST_WIDTH][DST_HEIGHT] = (float(*)[3][DST_WIDTH][DST_HEIGHT])ps_malloc(sizeof(float) * (DST_WIDTH * DST_HEIGHT) * 3);
  float(*output)[3087][85] = (float(*)[3087][85])ps_malloc(sizeof(float) * (3087 * 85));
  resizeImage(*picture_1, resizedImage);
  uint16_t(*resizedImage_2d)[DST_WIDTH][DST_HEIGHT] = (uint16_t(*)[DST_WIDTH][DST_HEIGHT])resizedImage;

  normalizeImage(*resizedImage_2d, normalized);
  Serial.println("passing ");
  unsigned long startTime = micros();
  forward_pass(normalized, output);
  unsigned long endTime = micros();
  Serial.println("passed ");
  unsigned long duration_us = endTime - startTime;  // Duration in microseconds
  float duration_ms = duration_us / 1000.0;         // Convert to milliseconds
  Serial.print("Execution time: ");
  Serial.print(duration_us);
  Serial.println(" microseconds");
  Serial.print("Execution time: ");
  Serial.print(duration_ms);
  Serial.println(" milliseconds");
  free(normalized);
  free(resizedImage);



  // Process detections
  Detection *detections = (Detection *)ps_malloc(sizeof(Detection) * 600);
  int det_count = 0;
  parse_yolo_output(*output, detections, &det_count);

  // Print final detections
  for (int i = 0; i < det_count; i++) {
    Serial.print("[ Detected Class ");
    Serial.print(" id = ");
    Serial.print(detections[i].class_id);
    Serial.print(" x= ");
    Serial.print(detections[i].x);
    Serial.print(" y= ");
    Serial.print(detections[i].y);
    Serial.print(" w =");
    Serial.print(detections[i].w);
    Serial.print(" h= ");
    Serial.print(detections[i].h);
    Serial.print(" conf= ");
    Serial.print(detections[i].confidence);
    Serial.println("]");
    tft.drawRect((int)detections[i].x, (int)detections[i].y, (int)detections[i].w, (int)detections[i].h, TFT_RED);
    tft.drawRect((int)detections[i].x - 1, (int)detections[i].y - 1, (int)detections[i].w + 2, (int)detections[i].h + 2, TFT_RED);
  }
}

void loop() {
}
