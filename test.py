import liveness_detection

detector_path = './detector'

# demo = liveness_detection.Demo('./output/AlexNetPro/Print/Print.model', './output/AlexNetPro/Print/train_le.pickle', detector_path, 0.8)
demo = liveness_detection.Demo('./output/AlexNet/Print/Print.model', './output/train_le.pickle', './detector', 0.8)
demo.start_detection()
