import tensorflow as tf
from sklearn.model_selection import train_test_split
from operator import itemgetter
#from tf_pose import common
from cv2 import rectangle
import numpy as np
import random
import pickle
import math
import os

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
        constx = box_size / 2
        consty = box_size / 2
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


def trackor_skeletons(npimg, currentFrame, trackers, humans):
    image_h, image_w = npimg.shape[:2]
    frame_skeletons = []
    frame_skeleton = []
    pontos = []
    contagem = []

    if len(humans) > 1:
        for c in range(len(trackers)):
            d_min = image_w ** 2
            Id_ord = -1
            a = 0
            centrox = trackers[c, 0] + (trackers[c, 2] - trackers[c, 0]) / 2
            centroy = trackers[c, 1] + (trackers[c, 3] - trackers[c, 1]) / 2
            # centrox = centrox / image_w
            # centroy = centroy / image_h
            for human in humans:
                if 0 not in human.body_parts.keys():
                    continue
                x1 = human.body_parts[0].x * image_w
                y1 = human.body_parts[0].y * image_h
                distancia = math.sqrt((x1 - centrox) ** 2 + (y1 - centroy) ** 2)
                if distancia < d_min:
                    d_min = distancia
                    Id_ord = a
                a += 1

            ponto = len(humans[Id_ord].body_parts.keys())
            pontos.append(ponto)
            # print(pontos)

            # for i in range(16):
            #     if i in humans[Id_ord].body_parts.keys():
            #         contagem.append(i)
            #         if len(contagem) == 16:
            #             frame_skeleton.append(currentFrame)
            #             frame_skeleton.append(int(trackers[c, 4]))
            #             for i in range(16):
            #                 x = humans[Id_ord].body_parts[i].x * image_w
            #                 y = humans[Id_ord].body_parts[i].y * image_h
            #                 frame_skeleton.append(x)
            #                 frame_skeleton.append(y)
            #             frame_skeletons.append(frame_skeleton)

            if pontos[c] == 17:
                frame_skeleton = []
                frame_skeleton.append(currentFrame)
                frame_skeleton.append(int(trackers[c, 4]))
                for i in range(16):
                    if i not in humans[Id_ord].body_parts.keys():
                        continue
                    x = humans[Id_ord].body_parts[i].x * image_w
                    y = humans[Id_ord].body_parts[i].y * image_h
                    frame_skeleton.append(x)
                    frame_skeleton.append(y)
                frame_skeletons.append(frame_skeleton)

            if pontos[c] == 18:
                frame_skeleton = []
                frame_skeleton.append(currentFrame)
                frame_skeleton.append(int(trackers[c, 4]))
                for i in range(16):
                    # if i == 15:
                    #     continue
                    x = humans[Id_ord].body_parts[i].x * image_w
                    y = humans[Id_ord].body_parts[i].y * image_h
                    frame_skeleton.append(x)
                    frame_skeleton.append(y)
                frame_skeletons.append(frame_skeleton)
    return (frame_skeletons)


def create_Dataset_Test(sequencia, nome, videos_sequencia_path):
    sequencia_esque = []
    coordenada = []
    sequencia_two_best = []
    distancia_acumulada = dict()
    DADOS = nome + '.dat'
    DADOS = os.path.join(videos_sequencia_path, nome + '.dat')
    for f in range(1, len(sequencia)):
        for s in range(len(sequencia[f])):
            id_atual = sequencia[f][s][1]
            id_anterior = -1
            for s_anterior in range(len(sequencia[f - 1])):
                if id_atual == sequencia[f - 1][s_anterior][1]:
                    pos_anterior = s_anterior
                    id_anterior = id_atual
                    break
            if id_anterior > 0:
                mao_esq_ant_x = sequencia[f - 1][pos_anterior][16]
                mao_esq_ant_y = sequencia[f - 1][pos_anterior][17]
                mao_dir_ant_x = sequencia[f - 1][pos_anterior][10]
                mao_dir_ant_y = sequencia[f - 1][pos_anterior][11]
                mao_esq_x = sequencia[f][s][16]
                mao_esq_y = sequencia[f][s][17]
                mao_dir_x = sequencia[f][s][10]
                mao_dir_y = sequencia[f][s][11]
                d_esq = math.sqrt((mao_esq_ant_x - mao_esq_x) ** 2 + (mao_esq_ant_y - mao_esq_y) ** 2)
                d_dir = math.sqrt((mao_dir_ant_x - mao_dir_x) ** 2 + (mao_dir_ant_y - mao_dir_y) ** 2)
                if not id_atual in distancia_acumulada:
                    distancia_acumulada[id_atual] = 0
                distancia_acumulada[id_atual] += d_esq + d_dir
    distancia_acumulada_ord = sorted(distancia_acumulada.items(), key=itemgetter(1), reverse=True)
    # print(distancia_acumulada_ord)
    id1 = distancia_acumulada_ord[0]
    b, a = id1
    id2 = distancia_acumulada_ord[1]
    d, c = id2

    for k in range(len(sequencia)):
        coordenadas = []
        if len(sequencia[k]) >= 2:
            for g in range(len(sequencia[k])):
                id_atual = sequencia[k][g][1]
                if id_atual == b or id_atual == d:
                    sequencia_two_best.append(sequencia[k][g])
                    for i in range(2, len(sequencia[k][g])):
                        coordenadas.append(sequencia[k][g][i])
            coordenadasss = np.array(coordenadas, dtype=float)
            if len(coordenadasss) == 68:
                coordenadasss = coordenadasss.reshape((1, 68, 1))
                coordenada.append(coordenadasss)

    for l in range(10):
        cord = coordenada[l]
        sequencia_esque.append(cord)
    with open(DADOS, 'wb') as f:
        pickle.dump(sequencia_esque, f, pickle.HIGHEST_PROTOCOL)

def create_Dataset(sequencia, nome, videos_sequencia_path, videos_seq_length, currentFrame, videos_sequencia_paths, videos_sequencia_path_2, videos_labels):
    sequencia_esque = []
    sequencia_esque2 = []
    coordenada = []
    sequencia_two_best = []
    distancia_acumulada = dict()
    DADOS = nome + '.dat'
    DADOS = os.path.join(videos_sequencia_path, nome + '.dat')
    DADOS2 = os.path.join(videos_sequencia_path_2, nome + '_2' + '.dat')
    for f in range(1, len(sequencia)):
        for s in range(len(sequencia[f])):
            id_atual = sequencia[f][s][1]
            id_anterior = -1
            for s_anterior in range(len(sequencia[f - 1])):
                if id_atual == sequencia[f - 1][s_anterior][1]:
                    pos_anterior = s_anterior
                    id_anterior = id_atual
                    break
            if id_anterior > 0:
                mao_esq_ant_x = sequencia[f - 1][pos_anterior][16]
                mao_esq_ant_y = sequencia[f - 1][pos_anterior][17]
                mao_dir_ant_x = sequencia[f - 1][pos_anterior][10]
                mao_dir_ant_y = sequencia[f - 1][pos_anterior][11]
                mao_esq_x = sequencia[f][s][16]
                mao_esq_y = sequencia[f][s][17]
                mao_dir_x = sequencia[f][s][10]
                mao_dir_y = sequencia[f][s][11]
                d_esq = math.sqrt((mao_esq_ant_x - mao_esq_x) ** 2 + (mao_esq_ant_y - mao_esq_y) ** 2)
                d_dir = math.sqrt((mao_dir_ant_x - mao_dir_x) ** 2 + (mao_dir_ant_y - mao_dir_y) ** 2)
                if not id_atual in distancia_acumulada:
                    distancia_acumulada[id_atual] = 0
                distancia_acumulada[id_atual] += d_esq + d_dir
    distancia_acumulada_ord = sorted(distancia_acumulada.items(), key=itemgetter(1), reverse=True)
    # print(distancia_acumulada_ord)
    id1 = distancia_acumulada_ord[0]
    b, a = id1
    id2 = distancia_acumulada_ord[1]
    d, c = id2

    for k in range(len(sequencia)):
        coordenadas = []
        coordenada_x = []
        coordenada_y = []
        if len(sequencia[k]) >= 2:
            for g in range(len(sequencia[k])):
                id_atual = sequencia[k][g][1]
                if id_atual == b or id_atual == d:
                    sequencia_two_best.append(sequencia[k][g])
                    for i in range(2, len(sequencia[k][g])):
                        coordenadas.append(sequencia[k][g][i])
            for j in range(len(coordenadas)):
                resto = j % 2
                if resto == 0:
                    coordenada_x.append(coordenadas[j])
                else:
                    coordenada_y.append(coordenadas[j])
            total_coordenada = coordenada_x + coordenada_y
            coordenadasss = np.array(total_coordenada, dtype=float)
            if len(coordenadasss) == 64:
                coordenadasss = coordenadasss.reshape((2, 32, 1))
                coordenada.append(coordenadasss)
    print(len(coordenada))
    for l in range(10):
        cord = coordenada[l]
        sequencia_esque.append(cord)
    with open(DADOS, 'wb') as f:
        pickle.dump(sequencia_esque, f, pickle.HIGHEST_PROTOCOL)

    if len(coordenada) >= 20:
        if not os.path.exists(videos_sequencia_path_2):
            os.makedirs(videos_sequencia_path_2)
        videos_seq_length.append(currentFrame)
        videos_sequencia_paths.append(videos_sequencia_path_2)
        if nome.startswith("fi"):
            videos_labels.append(1)
        elif nome.startswith("no"):
            videos_labels.append(0)
        for b in range(10, 20):
            sequencia_esque2.append(coordenada[b])
        with open(DADOS2, 'wb') as f:
            pickle.dump(sequencia_esque2, f, pickle.HIGHEST_PROTOCOL)


def data_generator(data_paths, labels, batch_size, seq_length):
    while True:
        indexes = np.arange(len(data_paths))
        np.random.shuffle(indexes)
        select_indexes = indexes[:batch_size]
        data_paths_batch = [data_paths[i] for i in select_indexes]
        labels_batch = [labels[i] for i in select_indexes]

        X, y = get_sequences(data_paths_batch, labels_batch, seq_length)

        yield X, y


def data_generator_files(data, labels, batch_size):
    while True:
        indexes = np.arange(len(data))
        np.random.shuffle(indexes)
        select_indexes = indexes[:batch_size]
        X = [data[i] for i in select_indexes]
        y = [labels[i] for i in select_indexes]
        yield X, y


# def get_sequences(data_paths, labels, seq_length):
#     X, y = [], []
#     for data_path, label in zip(data_paths, labels):
#         for filename in os.listdir(data_path):
#             sequencias_open = os.path.join(data_path, filename)
#             with open(sequencias_open, "rb") as f:
#                 sequencia_completa = pickle.load(f)
#             x = sequencia_completa
#             X.append(x)
#             y.append(label)
#     X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=seq_length, padding='pre', truncating='pre')
#     return np.array(X), np.array(y)

def get_sequences(data_paths, labels, seq_length):
    X, y = [], []
    for data_path, label in zip(data_paths, labels):
        for filename in os.listdir(data_path):
            sequencias_open = os.path.join(data_path, filename)
            sequencia_2 = []
            with open(sequencias_open, "rb") as f:
                sequencia_completa = pickle.load(f)
            for i in range(len(sequencia_completa)):
                nova_sequencia = []
                Seq = sequencia_completa[i]
                Seq_res = Seq.reshape(64,)
                Coordenadas_pontos = [1, 0, 14, 15, 1, 2, 3, 4, 1, 5, 6, 7, 1, 8, 9, 10, 1, 11, 12, 13]
                for k in range(4):
                    Const = 16*k
                    for j in range(20):
                        pontos = Coordenadas_pontos[j] + Const
                        Seq_2 = Seq_res[pontos]
                        nova_sequencia.append(Seq_2)
                        sequenciaa = np.array(nova_sequencia, dtype=float)
                Seq_res_2 = sequenciaa.reshape(4, 10, 2, order='F')
                sequencia_2.append(Seq_res_2)
            x = sequencia_2
            X.append(x)
            y.append(label)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=seq_length, padding='pre', truncating='pre')
    return np.array(X), np.array(y)


# def get_sequences(data_paths, labels, seq_length):
#     X, y = [], []
#     for data_path, label in zip(data_paths, labels):
#         for filename in os.listdir(data_path):
#             nova_sequencia = []
#             sequencias_open = os.path.join(data_path, filename)
#             with open(sequencias_open, "rb") as f:
#                 sequencia_completa = pickle.load(f)
#                 for i in range(len(sequencia_completa)):
#                     Seq = sequencia_completa[i]
#                     #Seq_res = Seq.reshape(64,)
#                     #Seq_res = Seq.reshape(64, 1, 1)
#                     #Seq_res = Seq.reshape(32, 1, 2, order='F')
#                     Seq_res = Seq.reshape(16, 2, 2, order='F')
#                     nova_sequencia.append(Seq_res)
#             x = nova_sequencia
#             X.append(x)
#             y.append(label)
#     X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=seq_length, padding='pre', truncating='pre')
#     return np.array(X), np.array(y)

def get_generators(videos_seq_length, videos_sequencia_paths, videos_labels, fix_len, batch_size):
    avg_length = int(float(sum(videos_seq_length)) / max(len(videos_seq_length), 1))

    train_path, test_path, train_y, test_y = train_test_split(videos_sequencia_paths, videos_labels, test_size=0.20,
                                                              random_state=42)
    train_path, valid_path, train_y, valid_y = train_test_split(train_path, train_y, test_size=0.20, random_state=42)

    if fix_len is not None:
        avg_length = fix_len

    len_train, len_valid = len(train_path), len(valid_path)

    train_gen = data_generator(train_path, train_y, batch_size, avg_length)

    validate_gen = data_generator(valid_path, valid_y, batch_size, avg_length)

    test_x, test_y = get_sequences(test_path, test_y, avg_length)

    treino_total_x, treino_total_y = get_sequences(videos_sequencia_paths, videos_labels, avg_length)

    return train_gen, validate_gen, test_x, test_y, avg_length, len_train, len_valid, treino_total_x, treino_total_y


# def get_generators(videos_seq_length, videos_sequencia_paths, videos_labels):
#     avg_length = int(float(sum(videos_seq_length)) / max(len(videos_seq_length), 1))
#
#     treino_total_x, treino_total_y = get_sequences(videos_sequencia_paths, videos_labels, avg_length)
#
#     return treino_total_x, treino_total_y
#
# def get_generators_2(videos_seq_length, data_train, target_train, batch_size):
#     avg_length = int(float(sum(videos_seq_length)) / max(len(videos_seq_length), 1))
#
#     train_path, valid_path, train_y, valid_y = train_test_split(data_train, target_train, test_size=0.20, random_state=42)
#
#     len_train, len_valid = len(data_train), len(valid_path)
#
#     train_gen = data_generator(data_train, target_train, batch_size, avg_length)
#
#     validate_gen = data_generator(valid_path, valid_y, batch_size, avg_length)
#
#     return train_gen, validate_gen, avg_length, len_train, len_valid