from skimage.transform import resize
from skimage.color import rgba2rgb, gray2rgb
import numpy as np

def classify_image_with_unknown(img, img_size, model, threshold=0.4):

    ##RGB conversion
    if len(img.shape) == 2:
        img = gray2rgb(img)
    elif img.shape[-1] == 4:
        img = rgba2rgb(img)

    #Prediction
    img = resize(img, img_size, anti_aliasing=True).flatten()
    probabilities = model.predict_proba(img.reshape(1, -1))
    max_confidence = np.max(probabilities)
    
    return model.predict(img.reshape(1, -1))[0] if max_confidence >= threshold else "unknown"

def test(image, img_size, frame_count, best_estimator):
    if image is not None and image.size  > 0:
        # Classify the image
        result = classify_image_with_unknown(image, img_size, best_estimator, threshold=0.4)
        print(f"Frame: {frame_count}, Classification Result: {result}")
        return result
    else:
        print(f"Skipping frame_{frame_count}: Not a useful frame.")