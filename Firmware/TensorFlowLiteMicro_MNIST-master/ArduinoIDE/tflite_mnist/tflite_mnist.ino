



#include <TensorFlowLite.h>


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"






#include "model.h"
#include "mnist.h"

#include <StopWatch.h>

#include "tensorflow/lite/micro/micro_profiler.h"
//#define USING_PICO

#define NDEBUG

StopWatch sw_millis;    // MILLIS (default)
StopWatch sw_micros(StopWatch::MICROS);
StopWatch sw_secs(StopWatch::SECONDS);
StopWatch SWarray[5];

int output_inference = 13;

// Globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* reporter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr long int kTensorArenaSize = 450000;  // Just pick a big enough number

uint8_t tensor_arena[kTensorArenaSize] = { 0 };
float* input_buffer = nullptr;
int test_image_cnt = 0;
int num_correct_cnt = 0;

// set up logging
//tflite::Profiler* profiler;

void bitmap_to_float_array(float* dest, const unsigned char* bitmap) {  // Populate input_vec with the monochrome 1bpp bitmap
  int pixel = 0;
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      int B = x / 8;  // the Byte # of the row
      int b = x % 8;  // the Bit # of the Byte
      dest[pixel] = (bitmap[y * 4 + B] >> (7 - b)) & 0x1 ? 1.0f : 0.0f;
      pixel++;
    }
  }
}

void print_input_buffer() {
  char output[28 * 29];  // Each row should end row newline
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      output[y * 29 + x] = input_buffer[y * 28 + x] > 0 ? ' ' : '#';
    }
    output[y * 29 + 28] = '\n';
  }
  reporter->Report(output);
}

void setup() {
  // Setup LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Load Model
  static tflite::MicroErrorReporter error_reporter;
  reporter = &error_reporter;


  reporter->Report("Let's use AI to recognize some numbers!");
  delay(1000);
  model = tflite::GetModel(tf_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    reporter->Report("Model is schema version: %d\nSupported schema version is: %d", model->version(), TFLITE_SCHEMA_VERSION);
    while (1) {}
    return;
  }

  // Setup our TF runner
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    reporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Save the input buffer to put our MNIST images into
  input_buffer = input->data.f;
  pinMode(output_inference, OUTPUT);

  // Stopwatch
  SWarray[0].start();

  delay(5000);  //Delay so we can start uart logger
  digitalWrite(LED_BUILTIN, HIGH);

  reporter->Report("Time Pass(1)/Fail(2) Number");
}

int is_passfail = 0;

void loop() {

  // Ensure Pin Low
  digitalWrite(output_inference, LOW);
  sw_micros.reset();

  // Number Images
  const int num_test_images = (sizeof(test_images) / sizeof(*test_images));

  //Input Test Model
  bitmap_to_float_array(input_buffer, test_images[test_image_cnt]);
  /**/

  // Run inference
  digitalWrite(output_inference, HIGH);
  sw_micros.start();

  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    reporter->Report("Invoke failed %d", invoke_status);
    return;
  }

  digitalWrite(output_inference, LOW);
  sw_micros.stop();

  // Get result
  float* result = output->data.f;
  int number = (std::distance(result, std::max_element(result, result + 10)));
  // Check if correct
  if (correct_number[test_image_cnt] == number) {
    num_correct_cnt++;
    is_passfail = 1;
  } else {
    is_passfail = 2;
  }

  //reporter->Report( "Image %d looks like the number: %d",test_image_cnt, std::distance( result, std::max_element( result, result + 10 ) ) );
  reporter->Report( "%d %d %d",sw_micros.elapsed(),is_passfail,number);
  //reporter->Report( "%d",profiler);
  //Serial.println(sw_micros.elapsed());

  // End Test
  if (++test_image_cnt >= num_test_images) {
    reporter->Report("Correct: %d OutOf: %d", num_correct_cnt, num_test_images);
    // Special Tests
    bitmap_to_float_array(input_buffer, test_images[101]);
    sw_micros.reset();
    sw_micros.start();
    invoke_status = interpreter->Invoke();
    sw_micros.stop();
    reporter->Report( "All_0s %d us",sw_micros.elapsed());

    bitmap_to_float_array(input_buffer, test_images[102]);
    sw_micros.reset();
    sw_micros.start();
    invoke_status = interpreter->Invoke();
    sw_micros.stop();
    reporter->Report( "All_1s %d us",sw_micros.elapsed());

    bitmap_to_float_array(input_buffer, test_images[103]);
    sw_micros.reset();
    sw_micros.start();
    invoke_status = interpreter->Invoke();
    sw_micros.stop();
    reporter->Report( "1s&0s %d us",sw_micros.elapsed());

    delay(10000);

    //Turn on LED when finished

    digitalWrite(LED_BUILTIN, LOW);

    test_image_cnt = 0;
    num_correct_cnt = 0;
    while (1) {};
  }

  // Wait 1-sec til before running again
  delay(200);
}
