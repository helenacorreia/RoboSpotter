import logging
import time

from pprint import pprint

import glob
import os
import common
from common import CocoPart
import cv2
import numpy as np
import PIL.Image as PImage
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import argparse

from tf_pose.estimator import TfPoseEstimator, PoseEstimator, BodyPart, Human
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.sort import Sort, parse_args, trackor_skeletons
from operator import itemgetter

from pafprocess import pafprocess

import math
import pickle 



DADOS = "pickle.dat"


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def nothing(x):
    pass

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def delet():
    path = "C:/Users/lenaa/Desktop/tf-pose-estimation/data/train/tfpose/det"
    dir = os.listdir(path)
    for file in dir:
        if file == "C:/Users/lenaa/Desktop/tf-pose-estimation/data/train/tfpose/det/det.txt":
            os.remove(file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='./images/video_curto.mp4')
    #parser.add_argument('--video', type=str, default='C:/Users/lenaa/Desktop/tf-pose-estimation/videos/traine/NonFight/_fPfNbHM16M_0.avi')
    
    #parser.add_argument('--camera', type=str, default=0)

    parser.add_argument('--resize', type=str, default='0x0', help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0, help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False, help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--tensorrt', type=str, default="False", help='for tensorrt process.')
    
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()
    phase = args.phase

#    if w > 0 and h > 0:
#        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
#    else:
#        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    w = 656
    h = 368
    
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(656, 368), trt_bool=str2bool(args.tensorrt))
    
    pasta = 'C:/Users/lenaa/Desktop/tf-pose-estimation/videos'
    #pasta = 'C:/Users/lenaa/OneDrive/Ambiente de Trabalho/ForTal/videos'


    for filename in os.listdir(pasta):
        if filename.endswith(".avi") or filename.endswith(".mp4"):
            video_file = os.path.join(pasta, filename[:-4] + filename[-4:])
            
            nome = filename[:-4]
            print(nome)
            
            cam = cv2.VideoCapture(video_file)
            
            if cam.isOpened() is False:
                print("Error opening video stream or file")
            
            currentFrame = 0
            new_humans = 0
            sequencia = []
            pontos = []
            while (cam.isOpened()):            
                ret_val, image = cam.read()
                        
                if ret_val == True:
                    human = Human([])
                    orange_color = (0,140,255)
                    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                    #image = TfPoseEstimator.draw_rectangle(image, humans, imgcopy=False)
                    #score = TfPoseEstimator.score(image, humans, imgcopy=False)
                    
                    pontos = os.path.join('C:/Users/lenaa/Desktop/tf-pose-estimation/videos', nome + '.txt')
                    
                    #Numero de pessoas na imagem 
                    num_people = len(humans)
                    
                    print(currentFrame)
                    
#                    for human in humans:
#                        ponto = len(human.body_parts.keys()) 
#                        arquivo = open(pontos, 'a+')
#                        arquivo.write(str(currentFrame) + ',' + str(ponto) + ',' + str(video_file) + '\n')
#                        arquivo.close()
                    
                                       
                    #Escreve no bloco de notas para o tracker
                    TfPoseEstimator.write_notepad(image, humans, currentFrame)
                    
                    #Inicialização do tracker
                    if len(humans) > 1:
                        if currentFrame >= 1:
                            if not os.path.exists('output'):
                                os.makedirs('output')
                            pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
                            for seq_dets_fn in glob.glob(pattern):
                                mot_tracker = Sort() #create instance of the SORT tracker
                                seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
                                seq = seq_dets_fn[pattern.find('*'):].split('/')[0]

                                
                                with open('C:/Users/lenaa/Desktop/tf-pose-estimation/output/tfpose/det/det.txt','w') as out_file:
                                    print("Processing %s."%(seq))
                                    if(len(humans) != 0):
                                        dets = seq_dets[seq_dets[:, 0]==currentFrame, 2:7]
                                        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                                        trackers = mot_tracker.update(dets)
                                        for d in trackers:
                                            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(currentFrame, d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
                                            cv2.putText(image, "id: %d" % (int(d[4])), (int(d[0]),int(d[1])),0, 1, (255,255,255),2)
                            
                                        frame_skeletons = trackor_skeletons(image, currentFrame, trackers, humans)
                                        if len(frame_skeletons) != 0:
                                            sequencia.append(frame_skeletons)
                                            print(sequencia)
            
                                        
            #                for f in range ( 1, len(sequencia)):
            #                    for s in range(len(sequencia[1])):
            #                        id_atual = sequencia[f][s][1]
            #                        id_anterior = -1
            #                        for s_anterior in range (len(sequencia[f-1])):
            #                            if id_atual == sequencia [f-1][s_anterior][1]:
            #                                pos_anterior = s_anterior
            #                                id_anterior = id_atual
            #                                break
            #                        if id_anterior > 0:
            #                            mao_esq_ant_x = sequencia[f-1][pos_anterior][16]
            #                            mao_esq_ant_y = sequencia[f-1][pos_anterior][17]
            #                            mao_dir_ant_x = sequencia[f-1][pos_anterior][10]
            #                            mao_dir_ant_y = sequencia[f-1][pos_anterior][11]
            #                            mao_esq_x = sequencia[f][s][16]
            #                            mao_esq_y = sequencia[f][s][17]
            #                            mao_dir_x = sequencia[f][s][10]
            #                            mao_dir_y = sequencia[f][s][11]
            #                            cv2.line(image, (int(mao_esq_ant_x), int(mao_esq_ant_y)), (int(mao_esq_x), int(mao_esq_y)), (0,255,0), 2)
            #                            cv2.line(image, (int(mao_dir_ant_x), int(mao_dir_ant_y)), (int(mao_dir_x), int(mao_dir_y)), (0,255,0), 2)
                    
                    currentFrame += 1
            
#                    cv2.putText(image,
#                                "People: %d" % (num_people),
#                                (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                (255, 255, 255), 2)
#                    
#                    cv2.putText(image,
#                                "FPS: %f" % (1.0 / (time.time() - fps_time)),
#                                (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                (0, 255, 0), 2)
                    cv2.imshow(nome, image)
                    fps_time = time.time()
                    if cv2.waitKey(1) == 27:
                        break
                else:
                    break
                
            cam.release()
            cv2.destroyAllWindows()
            
            print('Finished')
            if os.path.exists("C:/Users/lenaa/Desktop/tf-pose-estimation/data/train/tfpose/det/det.txt"):
                os.remove("C:/Users/lenaa/Desktop/tf-pose-estimation/data/train/tfpose/det/det.txt")
             
            print(sequencia)
            if len(sequencia) != 0:
                two_best = []
                distancia_acumulada = dict()
                for f in range ( 1, len(sequencia)):
                    for s in range(len(sequencia[f])):
                        id_atual = sequencia[f][s][1]
                        id_anterior = -1
                        for s_anterior in range (len(sequencia[f-1])):
                            if id_atual == sequencia [f-1][s_anterior][1]:
                                pos_anterior = s_anterior
                                id_anterior = id_atual
                                break
                        if id_anterior > 0:
                            mao_esq_ant_x = sequencia[f-1][pos_anterior][16]
                            mao_esq_ant_y = sequencia[f-1][pos_anterior][17]
                            mao_dir_ant_x = sequencia[f-1][pos_anterior][10]
                            mao_dir_ant_y = sequencia[f-1][pos_anterior][11]
                            mao_esq_x = sequencia[f][s][16]
                            mao_esq_y = sequencia[f][s][17]
                            mao_dir_x = sequencia[f][s][10]
                            mao_dir_y = sequencia[f][s][11]
                            d_esq = math.sqrt((mao_esq_ant_x - mao_esq_x)**2 + (mao_esq_ant_y - mao_esq_y)**2)
                            d_dir = math.sqrt((mao_dir_ant_x - mao_dir_x)**2 + (mao_dir_ant_y - mao_dir_y)**2)
                            if not id_atual in distancia_acumulada:
                                distancia_acumulada[id_atual] = 0 
                            distancia_acumulada[id_atual] += d_esq + d_dir
                distancia_acumulada_ord = sorted(distancia_acumulada.items(), key=itemgetter(1), reverse=True)
                print(nome)
                print(distancia_acumulada_ord)
                id1 = distancia_acumulada_ord[0]
                b, a = id1
                id2 = distancia_acumulada_ord[1]
                d, c = id2
                
                coordenadas = []
                sequencia_two_best = []
                arquivo = open('C:/Users/lenaa/Desktop/tf-pose-estimation/sequencia_two_best/sequencia.txt', 'w')
                for e in range (len(sequencia)):
                    for g in range (len(sequencia[e])):
                        id_atual = sequencia[e][g][1]
                        if id_atual == b or id_atual == d:
                            sequencia_two_best.append(sequencia[e][g])
                            for i in range(len(sequencia[e][g])-1):
                                arquivo.write( str(sequencia[e][g][i]) + ', ')
                            arquivo.write( str(sequencia[e][g][-1]) +'\n')
                arquivo.close()
                
                with open('C:/Users/lenaa/Desktop/tf-pose-estimation/sequencia_two_best/sequencia.txt', 'r') as arquivo:
                    for linha in arquivo:
                        colunas = linha.split(",")
                        for i in range(2, len(colunas)):
                            coordenadas.append(colunas[i])
                coordenada = np.array(coordenadas, dtype = float)
                print(coordenada)
                
#                with open(DADOS, 'wb') as f:
#                    pickle.dump(coordenada, f, pickle.HIGHEST_PROTOCOL)
#                with open(DADOS, "rb") as f:
#                    print(pickle.load(f))
                        