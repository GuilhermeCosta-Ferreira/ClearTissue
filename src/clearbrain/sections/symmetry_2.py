# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm



# ================================================================
# 1. Section: Functions
# ================================================================
def get_symmetry_axis(img):
    angle_list = np.linspace(0, 180, 50, endpoint=False)
    data = np.where(img > 0, 1, 0)

    center = np.mean(np.argwhere(data), axis=0)
    #center = (img.shape[0] // 2, img.shape[1]// 2)
    length = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))


    separability = []
    for angle in tqdm(angle_list):
        symmetry_rate = get_symmetry_rate(data, angle)
        separability.append(symmetry_rate)
    """
    diffs = np.abs(np.asarray(separability) - 0.5)
    print(diffs)
    """
    best_symmetry_angle = angle_list[np.argmax(separability)]

    angle=best_symmetry_angle
    return add_line(img, angle, center, length, value=-10)



def get_symmetry_rate(img: np.ndarray, angle: float) -> float:
    center = (img.shape[0] // 2, img.shape[1]// 2)
    line_m, line_b = get_line_eq(angle, center)

    mask = side_mask(img.shape, angle)
    points = np.argwhere(mask == 1)

    """
    target_points = np.argwhere(~mask)
    plt.figure()
    plt.imshow(img.T, origin='lower')
    plt.imshow(mask.T, alpha=0.5, origin='lower', cmap="Blues")
    """

    sym_scores = []
    for point in points:
        """
        plt.figure()
        plt.imshow(img.T, origin='lower')
        plt.imshow(mask.T, alpha=0.5, origin='lower', cmap="Blues")

        x=np.arange(0, img.shape[0])
        y = line_m * x + line_b
        plt.plot(x, y, color="white", linewidth=2)
        """

        point_m, point_b = get_point_line(point, line_m)
        """
        print(line_m, line_b)
        print("point line")
        print(point_m, point_b)
        y = point_m * x + point_b
        plt.plot(x, y, color="red", linewidth=2, alpha=0.5)
        """

        intercept = get_line_intercept(
            (line_m, line_b),
            (point_m, point_b)
        )

        displacement_vector = np.array([intercept[0] - point[0], intercept[1] - point[1]])

        sym_point = (np.array(intercept)) + displacement_vector
        sym_point = np.round(sym_point, 0).astype(int)



        """
        print(f"point: {point}")
        print(f"intercept {intercept}")
        print(f"displacement_vector: {displacement_vector}")
        print(f"sym_point: {sym_point}")
        plt.scatter(point[0], point[1], color='red', label="point")
        #plt.scatter(sym_point[0], sym_point[1], color='blue', label="sym_point")
        plt.scatter(intercept[0], intercept[1], color='yellow', label="intercept")
        """

        """
        plt.quiver(
            intercept[1], intercept[0],
            displacement_vector[1], displacement_vector[0],
            color="black",
            scale_units="xy",
            scale=1,
        )
        plt.xlim((0, img.shape[0]))
        plt.ylim((0, img.shape[1]))
        plt.show()
        """



        if not (0 <= sym_point[0] < img.shape[0]) or not (0<= sym_point[1] < img.shape[1]):
            #print(f"Failed the sym for point {point}, got {sym_point}")
            continue

        """
        if not any(np.array_equal(sym_point, p) for p in target_points):
            print(f"Failed the sym for point {point}, got {sym_point}")
            #plt.scatter(sym_point[0], sym_point[1], color='blue', label="sym_point")
            continue
        else:
            #print(f"Found {sym_point} at target_points")
            plt.scatter(sym_point[0], sym_point[1], color='blue', label="sym_point")
        """


        sym_value = 1 if img[point[0], point[1]] == img[sym_point[0], sym_point[1]] else 0
        sym_scores.append(sym_value)



    max_sym_value = len(points)
    sym_rate = np.sum(sym_scores) / max_sym_value
    #print(sym_rate)
    #plt.show()

    return sym_rate

def add_line(img, angle_deg, center, length, value=1):
    h, w = img.shape
    y0, x0 = center

    theta = np.deg2rad(angle_deg)

    # image coords: x right, y down
    dx = np.cos(theta)
    dy = np.sin(theta)

    t = np.arange(-length, length + 1)

    x = np.round(x0 + t * dx).astype(int)
    y = np.round(y0 + t * dy).astype(int)

    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    line = np.zeros_like(img)
    line = img
    line[y[valid], x[valid]] = value

    return line


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_line_eq(angle: float, center: tuple | np.ndarray) -> tuple[float, float]:
    theta = np.deg2rad(alpha_to_theta(angle))
    m = np.tan(theta)
    b = center[1] - center[0] * m

    return m, b

def get_point_line(point: tuple | np.ndarray, line_m: float) -> tuple[float, float]:
    m = (-1) / line_m if line_m != 0 else 10e18
    b = point[1] - m * point[0]

    return m, b

def get_line_intercept(line_eq: tuple, point_eq: tuple) -> tuple[float, float]:

    x = (point_eq[1] - line_eq[1]) / (line_eq[0] - point_eq[0])
    if(line_eq[0] > 1e5):
        y = point_eq[0] * x + point_eq[1]
    else:
        y = line_eq[0] * x + line_eq[1]

    return (x, y)


def alpha_to_theta(alpha: float) -> float:
    if alpha > 180 or alpha < 0:
        raise ValueError(f"Alha can not be outside [0, 180], currently is {np.round(alpha, 2)}")
    if 0 <= alpha <= 90:
        return alpha
    else:
        return alpha - 180

def side_mask(shape: tuple | np.ndarray, angle_deg: float, eps: float = 0.5) -> np.ndarray:
    w, h = shape
    x0, y0 = (w // 2, h // 2)

    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)

    x, y = np.mgrid[0:w, 0:h]

    # signed side test
    signed = dx * (y - y0) - dy * (x - x0)

    return signed > eps
