from config import *
import logging

logging.TRACE = 5  # Custom log level for TRACE
logging.addLevelName(logging.TRACE, "TRACE")
logging.basicConfig(level=LOGGING_LEVEL, format='[%(levelname)s] %(name)s - %(funcName)s : %(message)s')

class CustomFormatter(logging.Formatter):
    """A custom logging formatter that adds color to log messages based on their level."""
    RED = "\x1b[31;1m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    WHITE = "\x1b[37;20m"
    ORANGE = "\x1b[38;5;208m"
    RESET = "\x1b[0m"

    FORMAT = '[%(levelname)s] %(name)s - %(funcName)s : %(message)s'

    def __init__(self, fmt=FORMAT):
        """ Initializes the CustomFormatter with the specified format.  """
        super().__init__(fmt=fmt)

        self.FORMATS = {
            logging.TRACE: self.WHITE + self.FORMAT + self.RESET,
            logging.DEBUG: self.WHITE + self.FORMAT + self.RESET,
            logging.INFO: self.BLUE + self.FORMAT + self.RESET,
            logging.WARNING: self.ORANGE + self.FORMAT + self.RESET,
            logging.ERROR: self.RED + self.FORMAT + self.RESET,
            logging.CRITICAL: self.RED + self.FORMAT + self.RESET
        }

    def format(self, record):
        """ Format the log record with the appropriate color based on its level. """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class CustomLogger(logging.Logger):
    """ A custom logger that supports the TRACE log level. """

    def trace(self, msg, *args, **kwargs):
        """ Log a message with level TRACE. """
        if self.isEnabledFor(logging.TRACE):
            self._log(logging.TRACE, msg, args, **kwargs)

def get_logger(name='logger'):
    """ Create and return a logger with the specified name and level. """

    logger = CustomLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    logger.propagate = False  

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = CustomFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

class Point:
    """ ======= A class representing a point in 2D space ======= """
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def to_tuple(self) -> tuple:
        return (self.x, self.y)

    @staticmethod
    def from_tuple(t: tuple) -> 'Point':
        return Point(t[0], t[1]) if len(t) == 2 else Point(0, 0)

class Box:
    """ ======= A class representing a quadrilateral defined by four points ======= """
    def __init__(self, top_left: Point, bottom_right: Point, \
        top_right: Point, bottom_left: Point):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.top_right = top_right
        self.bottom_left = bottom_left
        
        self.lines = [
            Line(top_left, top_right),
            Line(top_right, bottom_right),
            Line(bottom_right, bottom_left),
            Line(bottom_left, top_left)
        ]
        
    def check_point_inside(self, point: Point) -> bool:
        """
        Check if a point is inside the quadrilateral defined by the Box's four points.
        This implementation uses the winding number algorithm (or ray casting).
        It's robust for convex and concave polygons.
        """
        n = len(self.lines)
        inside = False

        p_x, p_y = point.x, point.y

        for i in range(n):
            current_line = self.lines[i]
            next_line = self.lines[(i + 1) % n] # Get the next vertex (wraps around)

            # Check if the point is on the horizontal line segment (edge parallel to x-axis)
            if current_line.start.y == next_line.start.y == p_y and min(current_line.start.x, next_line.start.x) <= p_x <= max(current_line.start.x, next_line.start.x):
                return True

            # Ray casting algorithm
            if ((current_line.start.y <= p_y < next_line.start.y) or (next_line.start.y <= p_y < current_line.start.y)) and \
               (p_x < (next_line.start.x - current_line.start.x) * (p_y - current_line.start.y) / (next_line.start.y - current_line.start.y) + current_line.start.x):
                inside = not inside

        return inside
    
class Rectangle:
    """ ======= A class representing a rectangle defined by two points: top-left and bottom-right ======= """
    def __init__(self, top_left: Point, bottom_right: Point):
        self.top_left = top_left
        self.bottom_right = bottom_right
        
    def get_center(self) -> Point:
        center_x = (self.top_left.x + self.bottom_right.x) / 2
        center_y = (self.top_left.y + self.bottom_right.y) / 2
        return Point(center_x, center_y)

    def __repr__(self):
        return f"Rectangle(top_left={self.top_left}, bottom_right={self.bottom_right})"

    def to_tuple(self) -> tuple:
        return (self.top_left.to_tuple(), self.bottom_right.to_tuple())

    @staticmethod
    def from_tuple(t: tuple) -> 'Rectangle':
        if len(t) == 2:
            return Rectangle(Point.from_tuple(t[0]), Point.from_tuple(t[1]))
        elif len(t) == 4:
            return Rectangle(Point(t[0], t[1]), Point(t[2], t[3]))
        else:
            print("Invalid tuple length for Rectangle, returning default Rectangle (0, 0) to (0, 0)")
            return Rectangle(Point(0, 0), Point(0, 0))

class Line:
    """ ======= A class representing a line defined by two points: start and end ======= """
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Line(start={self.start}, end={self.end})"

    def to_tuple(self) -> tuple:
        return (self.start.to_tuple(), self.end.to_tuple())

    @staticmethod
    def from_tuple(t: tuple) -> 'Line':
        return Line(Point.from_tuple(t[0]), Point.from_tuple(t[1])) if len(t) == 2 else Line(Point(0, 0), Point(0, 0))
