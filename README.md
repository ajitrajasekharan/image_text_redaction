# Image text redaction
Evaluation of  models for image text detection, recognition, and redaction. The evaluated models can detect horizontal print and handwritten text. Detection of curved text is beyond scope of these models although angled text in some orientations can be detected by one or mode or the models. The redaction module is a dummy placeholder that redacts any text that is recognized. It needs to be updated to selectively redact text based on use case.

Three models evaluated 
- [easy ocr](https://github.com/JaidedAI/EasyOCR)
- [TR OCR](https://github.com/microsoft/unilm/tree/master/trocr) It is also available on [Huggingface](https://huggingface.co/microsoft/trocr-base-handwritten)
- [Paddle OCR](https://github.com/PaddlePaddle/PaddleOCR)

_For TR OCR, both handwritten and print text models are tested_



<img src="gogle_image_example.png" width="800">
[Image from Google Healthare page](https://cloud.google.com/architecture/de-identification-of-medical-images-through-the-cloud-healthcare-api)
