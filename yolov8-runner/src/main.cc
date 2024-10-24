#include <iostream>
#include <string>
#include <assert.h>
#include <node_api.h>
#include <stdio.h>
#include "yolov8.hpp"
#include "opencv2/opencv.hpp"

YOLOv8* model = nullptr;

static napi_value WarmupModel(napi_env env, napi_callback_info info) {
  napi_status status;

  size_t argc = 2;
  napi_value args[2];
  status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
  assert(status == napi_ok);

  if (argc < 1) {
    napi_throw_type_error(env, NULL, "Wrong number of arguments");
    return NULL;
  }

  napi_valuetype valuetype;
  status = napi_typeof(env, args[0], &valuetype);
  assert(status == napi_ok);

  if (valuetype != napi_string) {
    napi_throw_type_error(env, NULL, "Wrong arguments");
    return NULL;
  }

  char enginePath[100];
  size_t pathLength = 0;
  status = napi_get_value_string_utf8(env, args[0], enginePath, 100, &pathLength);
  assert(status == napi_ok);

  if(model != nullptr) {
    delete model;
  }
  auto path = std::string(enginePath);
  std::cout << "Path is: " << enginePath << std::endl;
  model = new YOLOv8(path);
  model->make_pipe(true);
  
  assert(status == napi_ok);

  return napi_value{};
}

static napi_value RunInference(napi_env env, napi_callback_info info) {
  napi_status status;

  if (model == nullptr) {
    napi_throw_error(env, NULL, "NO_MODEL");
    return NULL;
  }

  size_t argc = 2;
  napi_value args[2];
  status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
  assert(status == napi_ok);

  if (argc < 1) {
    napi_throw_type_error(env, NULL, "Wrong number of arguments");
    return NULL;
  }

  napi_valuetype valuetype;
  status = napi_typeof(env, args[0], &valuetype);
  assert(status == napi_ok);

  if (valuetype != napi_object) {
    napi_throw_type_error(env, NULL, "Wrong arguments");
    return NULL;
  }
  void *imageBuffer;
  size_t bufSize;
  status = napi_get_buffer_info(env, args[0], &imageBuffer, &bufSize);
  assert(status == napi_ok);
  char* buf = (char*)imageBuffer;
  cv::Mat mat{1, (int)bufSize, CV_8UC1, buf};
  mat = cv::imdecode(mat, 1);

  // cv::Size size{640, 640};
  model->copy_from_Mat(mat);
  model->infer();
  std::vector<DetectObject> objs;
  model->detectPostprocess(objs);
  // std::cout << "Number of detected objects: " << objs.size() << std::endl;

  napi_value objArray;
  status = napi_create_array(env, &objArray);
  assert(status == napi_ok);
  napi_value pushFunc;
  status = napi_get_named_property(env, objArray, "push", &pushFunc);
  assert(status == napi_ok);
  (status == napi_ok);
  for(DetectObject obj : objs) {
    napi_value object;
    status = napi_create_object(env, &object);
    assert(status == napi_ok);
    
    napi_value label;
    status = napi_create_int32(env, obj.label, &label);
    assert(status == napi_ok);
    status = napi_set_named_property(env, object, "label", label);
    assert(status == napi_ok);
    
    napi_value prob;
    status = napi_create_double(env, (double)obj.prob, &prob);
    assert(status == napi_ok);
    status = napi_set_named_property(env, object, "prob", prob);
    assert(status == napi_ok);
    
    napi_value x;
    status = napi_create_double(env, (double)obj.rect.x, &x);
    assert(status == napi_ok);
    status = napi_set_named_property(env, object, "x", x);
    assert(status == napi_ok);
    
    napi_value y;
    status = napi_create_double(env, (double)obj.rect.y, &y);
    assert(status == napi_ok);
    status = napi_set_named_property(env, object, "y", y);
    assert(status == napi_ok);
    
    napi_value height;
    status = napi_create_double(env, (double)obj.rect.height, &height);
    assert(status == napi_ok);
    status = napi_set_named_property(env, object, "height", height);
    assert(status == napi_ok);
    
    napi_value width;
    status = napi_create_double(env, (double)obj.rect.width, &width);
    assert(status == napi_ok);
    status = napi_set_named_property(env, object, "width", width);
    assert(status == napi_ok);

    napi_value result;
    status = napi_call_function(env, objArray, pushFunc, 1, {&object}, &result);
    assert(status == napi_ok);
  }

  return objArray;
}

#define DECLARE_NAPI_METHOD(name, func)                                        \
  { name, 0, func, 0, 0, 0, napi_default, 0 }

napi_value Init(napi_env env, napi_value exports) {
  napi_status status;
  napi_property_descriptor descriptors[] = {DECLARE_NAPI_METHOD("warmupModel", WarmupModel), DECLARE_NAPI_METHOD("inference", RunInference)};
  // napi_property_descriptor addDescriptor = ;
  status = napi_define_properties(env, exports, 2, descriptors);
  assert(status == napi_ok);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)