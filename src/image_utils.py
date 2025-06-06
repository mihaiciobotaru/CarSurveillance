from config import *
from utils import Point, Box, Rectangle, Line
import cv2
import numpy as np

class ImageUtils:
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate an image by a given angle."""
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image
    
    @staticmethod
    def rotate_point_against_image(point: Point, image: np.ndarray, angle: float) -> Point:
        """Rotate a point against the center of the image by a given angle."""
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        point_np = np.array([[[point.x, point.y]]], dtype=np.float32)
        rotated_point_np = cv2.transform(point_np, rotation_matrix)
        
        return Point(int(rotated_point_np[0][0][0]), int(rotated_point_np[0][0][1]))
    
    @staticmethod
    def get_edges(image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection to an image."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 50, 500)
        return edges

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load an image from the specified path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return image
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, width: int=None, height: int=None):
        """
        Resizes an image while maintaining its aspect ratio.

        Args:
            image: The input image.
            width: The desired width of the resized image.
            height: The desired height of the resized image.

        Returns:
            The resized image.
        """
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def display(image: np.ndarray|str, title: str = "Image", display=True, size: tuple[int, int]|int = 1000) -> None:
        """Display an image in a window."""
        
        if display is False:
            return
        
        if isinstance(image, str): # image variable is a path to an image file; we need to load it
            try:
                image = ImageUtils.load_image(image)
            except ValueError as e:
                print(e)
                return

        try:
            if isinstance(size, int):
                image = ImageUtils.resize_with_aspect_ratio(image, width=size)
            elif isinstance(size, tuple) and len(size) == 2:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            
            cv2.imshow(title, image)
            while True: # pressing ESC or closing the window will exit the loop and stop displaying the image
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error displaying image: {e}")
            return

    @staticmethod
    def draw_line_on_image(image: np.ndarray, line: Line) -> np.ndarray:
        """Draw a line on the image."""
        if line.start and line.end:
            cv2.line(image, line.start.to_tuple(), line.end.to_tuple(), (0, 255, 0), 2)
        return image
    
    @staticmethod  
    def draw_box_on_image(image: np.ndarray, box: Box) -> np.ndarray:
        """Draw a box on the image."""
        ImageUtils.draw_line_on_image(image, Line(box.top_left, box.top_right))
        ImageUtils.draw_line_on_image(image, Line(box.top_right, box.bottom_right))
        ImageUtils.draw_line_on_image(image, Line(box.bottom_right, box.bottom_left))
        ImageUtils.draw_line_on_image(image, Line(box.bottom_left, box.top_left))
        return image
    
    @staticmethod
    def draw_rectangle_on_image(image: np.ndarray, rectangle: Rectangle, text:str='') -> np.ndarray:
        """Draw a rectangle on the image."""
        cv2.rectangle(image, rectangle.top_left.to_tuple(), rectangle.bottom_right.to_tuple(), (0, 255, 0), 2)
        if len(text) > 0:
            cv2.putText(image, text, (rectangle.top_left.x + 10, rectangle.top_left.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    @staticmethod
    def draw_point_on_image(image: np.ndarray, point: Point, text:str='') -> np.ndarray:
        """Draw a point on the image."""
        cv2.circle(image, point.to_tuple(), 5, (0, 255, 0), -1)
        if len(text) > 0:
            cv2.putText(image, text, (point.x + 10, point.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image


def main() -> int:
    ImageUtils.display(SELECTED_IMAGE, "Example Image")

    return 0

if __name__ == "__main__":
    exit(main())
