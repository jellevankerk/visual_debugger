import cv2
import numpy as np

def show_image(original_function):
    """ Show a np.ndarray as using cv2.imshow"""
    def wrapper_function(*args, **kwargs):
        result = original_function(*args, **kwargs)
        
        if isinstance(result, np.ndarray):
            print(f"resulting array is {result.shape}")
            
            try:     
                cv2.imshow(original_function.__name__, result)
                cv2.waitKey(0)
            except Exception as e:
                print(f"image could be not shown as {original_function.__name__}")
                print(e)
        return result

    return wrapper_function


@show_image
def load_image(image_path):
    # load as a grayscale image
    return cv2.imread(image_path)

@show_image
def blur_image(image):
    return cv2.GaussianBlur(image,(5,5),0) 

@show_image
def edge_detection(image):
    return  cv2.Canny(image, 50, 150)

@show_image
def morph_close(edges):
    # Apply a morphological operation to close gaps in between edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

@show_image
def morph_dilate(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    return dilated

@show_image
def segment_image(image, closed):
    # Perform contour detection
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black mask for the segmented regions
    mask = np.zeros_like(image)

    # Iterate over detected contours and draw them on the mask
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # Apply the mask to the original image
    return  cv2.bitwise_and(image, mask)



if __name__ == "__main__":
    image = load_image(image_path=r'lenna.png')
    image = blur_image(image)
    edges = edge_detection(image)
    
    closed = morph_close(edges)
    segment_image(image, closed)
    
    dilated = morph_dilate(edges)
    segment_image(image, dilated)
    