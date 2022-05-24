
import cv2
import os


def read_and_down(file_path, down_factor=32):
    '''
    For read and down scale the images.
    :param file_path: the path for original images
    :param down_factor: the scale down factor for original image
    :return: src: original image; dwn: scaled image
    '''
    src = cv2.imread(file_path)
    width = int(src.shape[1] // down_factor)
    height = int(src.shape[0] // down_factor)
    dim = (width, height)
    dwn = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
    dwn = cv2.cvtColor(dwn, cv2.COLOR_BGR2GRAY)
    #dwn = cv2.GaussianBlur(dwn, (3,3), 0)
    return src, dwn


def axis_candidate(projection):
    # Assumed separation: 12*5
    threshold = projection.max() * 0.5
    axis_candidates = []
    for p in range(0, projection.size):
        if projection[p] <= threshold:
            axis_candidates.append(p)
    return axis_candidates


def clean_candidates(candidates, projections, if_x=1):
    axis_breaks = []
    if if_x == 1:
        axis_factor = 12
    else:
        axis_factor = 5
    break_length = projections.size / axis_factor
    for c in range(1, len(candidates)):
        if candidates[c] - candidates[c - 1] > break_length / 3:
            axis_breaks.append(candidates[c - 1])
            axis_breaks.append(candidates[c])
    return axis_breaks


def grid_crop(x_cand, y_cand, trgt, file_string, scale):
    m = len(x_cand)
    n = len(y_cand)
    sub_images = []
    # error checking
    print(m*n)
    for y in range(0, n - 1, 2):
        for x in range(0, m - 1, 2):
            sub_images.append(trgt[y_cand[y] * scale:y_cand[y + 1] * scale, x_cand[x] * scale:x_cand[x + 1] * scale])
    p = 1
    for img in sub_images:
        fs = file_string + str(p) + '.tif'
        cv2.imwrite(fs, img)
        p += 1


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def sep_image(img_file, img_folder, thr_value=160, scale=32):
    img_root = ''.join(img_file.split(".")[:-1])

    root_folder = os.path.join(img_folder, img_root)
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)
    # else:
    #     # already cropped
    #     return

    img, scaled = read_and_down(os.path.join(img_folder, img_file), scale)

    thr = cv2.threshold(scaled, thr_value, 255, cv2.THRESH_BINARY_INV)[1]
    h_proj = thr.sum(axis=1)
    w_proj = thr.sum(axis=0)
    cnd_x = clean_candidates(axis_candidate(w_proj), w_proj, 1)
    cnd_y = clean_candidates(axis_candidate(h_proj), h_proj, 0)

    file_string = os.path.join(root_folder, (img_root+'_grid_'))

    grid_crop(cnd_x, cnd_y, img, file_string, scale)


if __name__=='__main__':
    img_folder = 'E:/HKU_Study/PhD/Lab_work/Keyence_Images'
    for file in files(img_folder):
        print(file)
        sep_image(file, img_folder, 160, 16)