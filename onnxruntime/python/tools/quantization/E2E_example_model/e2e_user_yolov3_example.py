import os
from onnxruntime.quantization import get_calibrator, YoloV3DataReader, YoloV3VisionDataReader, YoloV3Evaluator, generate_calibration_table

def get_prediction_evaluation(model_path, augmented_model_path, validation_dataset, providers):
    image_list = os.listdir(validation_dataset)
    stride = 1000
    
    evaluator = None
    results = []
    for i in range(0, len(image_list), stride):
        print("Total %s images\nStart to process from %s with stride %s ..." % (str(len(image_list)), str(i), str(stride)))
        dr = YoloV3DataReader(validation_dataset, augmented_model_path=model_path, start_index=i, size_limit=stride, is_validation=True)
        evaluator = YoloV3Evaluator(model_path, dr, providers=providers)

        evaluator.predict()
        results += evaluator.get_result()[0]

    print("Total %s bounding boxes." % (len(results)))
        
    if evaluator:
        annotations = './annotations/instances_val2017.json'
        evaluator.evaluate(results, annotations)


def get_calibration_table(model_path, augmented_model_path, calibration_dataset):
    data_reader = YoloV3DataReader(calibration_dataset, augmented_model_path=augmented_model_path)
    # data_reader = YoloV3VisionDataReader(calibration_dataset, augmented_model_path=augmented_model_path)
    generate_calibration_table(model_path, augmented_model_path, data_reader, calibration_dataset=calibration_dataset)


if __name__ == '__main__':

    model_path = 'yolov3_new.onnx'
    # model_path = 'yolov3_merge_coco_openimage_500200_288x512_batch_nms_obj_300_score_0p35_iou_0p35_shape.onnx'
    # model_path = 'yolov3_merge_coco_openimage_500200_384x608_batch_nms_obj_300_score_0p35_iou_0p35_shape.onnx'
    augmented_model_path = 'augmented_model.onnx'
    # calibration_dataset = './val2017'
    calibration_dataset = './test2017'
    validation_dataset = './val2017'

    get_calibration_table(model_path, augmented_model_path, calibration_dataset)
    get_prediction_evaluation(model_path, augmented_model_path, validation_dataset, ["CUDAExecutionProvider"])



