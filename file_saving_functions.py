from cv2 import rectangle
from tf_pose import common
import numpy as np

box_size = 100

def draw_rectangle(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    pontos1 = {}
    pontos2 = {}
    for human in humans:
        # draw point
        for i in range(common.CocoPart.Neck.value):
            if i not in human.body_parts.keys():
                continue

            x = human.body_parts[i].x * image_w + 0.5
            y = human.body_parts[i].y * image_h + 0.5
            constx = 50
            consty = 50
            x1 = int(x - constx)
            y1 = int(y - consty)
            x2 = int(x + constx)
            y2 = int(y + consty)
            ponto1 = (x1, y1)
            ponto2 = (x2, y2)
            pontos1[i] = ponto1
            pontos2[i] = ponto2
            rectangle(npimg, ponto1, ponto2, (255, 0, 0), 2)
    return npimg


def find_point(image, humans, p):
    for human in humans:
        body_part = human.body_parts[p]
        x = int(body_part.x * image.shape[1] + 0.5)
        y = int(body_part.y * image.shape[0] + 0.5)
    return (x, y)


def write_notepad(npimg, humans, currentFrame, det_file_path):
    image_h, image_w = npimg.shape[:2]
    coordenadas = []
    arquivo = open(det_file_path, 'a+')
    for human in humans:
        if 0 not in human.body_parts.keys():
            continue
        x = human.body_parts[0].x * image_w + 0.5
        y = human.body_parts[0].y * image_h + 0.5
        constx = box_size/2
        consty = box_size/2
        x1 = int(x - constx)
        y1 = int(y - consty)
        x2 = int(x + constx)
        y2 = int(y + consty)
        ponto1 = (x1, y1)
        ponto2 = (x2, y2)
        rectangle(npimg, ponto1, ponto2, (255, 0, 0), 2)
        coordenadas = [x1, y1, x2, y2, 1]
        arquivo.write(
            str(currentFrame) + ',' + '-1' + ',' +
            str(coordenadas[0]) + ',' + str(coordenadas[1]) + ',' +
            str(coordenadas[2]) + ',' + str(coordenadas[3]) + ',' +
            '1' + ',' + '-1' + ',' + '-1' + ',' + '-1' + '\n')
    arquivo.close()