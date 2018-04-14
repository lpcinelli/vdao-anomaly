import numpy as np
import os
import random as rd
import math
import time
import re
import gc
import h5py
#from resnet50_altered import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, AveragePooling2D, Dropout
from keras.optimizers import Adamax, Adam
#from imagenet_utils import preprocess_input
from keras import backend as K

import datasets.hdf5_vdao as vdao

### MACROS ###

# BLOCOS DE TESTE -- Como nao e possivel treinar com os mesmo objetos do teste, foram feitos
# blocos de teste, onde a rede e treinada com todos os objetos exceto os que estao no bloco
# (Ex: Bloco de teste [0,2,5,6] - treino com [1,3,4,7,8,9,10,11])
VDAO_TEST_ENTRIES_OBJ_NB = [
    np.array([0, 2, 5, 6]),
    np.array([1, 3, 4, 8]),
    np.array([7, 9, 12, 13]),
    np.array([10, 11, 14, 19]),
    np.array([15, 16, 20, 22]),
    np.array([17, 21, 24, 29]),
    np.array([18, 28, 33, 38]),
    np.array([23, 25, 40, 41]),
    np.array([26, 27, 34, 39]),
    np.array([30, 35, 36, 46]),
    np.array([31, 37, 43, 49]),
    np.array([32, 42, 44, 47]),
    np.array([45, 50, 52, 56]),
    np.array([48, 51, 54, 55]),
    np.array([53, 56, 8, 10]),
    np.array([57, 11, 18, 24]),
    np.array([58, 13, 23, 46])
]

# Nomes das camadas onde foram extraidos os mapas de features
LAYER_NAME = [
    'res2a', 'res2b', 'res2c', 'res3a', 'res3b', 'res3c', 'res3d', 'res4a',
    'res4b', 'res4c', 'res4d', 'res4e', 'res4f', 'res5a', 'res5b', 'res5c',
    'avg_pool'
]

LAYER_NAME = [
    name + '_branch2a' for name in LAYER_NAME if name.startswith('res')
]

# lista com a quantidade de neurônios para cada camada escondida (2 camadas escondidas)
lst_2_lyr_top = np.array([1600])
lst_2_lyr_mid = np.array([50])

# lista com a quantidade de neurônios para cada camada escondida (2 camadas escondidas)
lst_1_lyr_top = np.array([1600])

# Determina qual modelo de  Fully Connected que sera treinado
# countList = 0 -- Sem camadas escondidas
# countList = 1 -- 1 camada  escondida
# countList = 2 -- 2 camadas  escondidas
# O codigo foi feito para se iterar nos 3 tipos de configuracao (while (countList < 3))
countList = 2

################################################

# changeModel -- seleciona o modelo de Fully Connected que é usado no treino


def main_prog():

    global pwd
    global mv_res_dir
    global countList
    global models_dir

    ##### VARIAVEIS TAMBEM POSSIVEIS DE SE ALTERAR #########

    # parte do nome dos .csv a serem salvos, usado para identificar qual o tipo do treinamento
    str_objs = 'ARTICLE_NEW_TEST_'

    # quantidade maxima de epocas
    EPS_MAX_RNG = 20

    # tamanho do batch para o 'fit'
    batch_max_val = 128

    ########################################################

    # ctr_int -- ITERA SOBRE AS 17 CAMADAS
    for out_layer in LAYER_NAME:
        for cnt_tst in range(len(VDAO_TEST_ENTRIES_OBJ_NB)):

            X_train, y_train, X_test, y_test, X_val, y_val = loadHDF5Batches(
                out_layer, cnt_tst)
            trn_shp = X_train.shape

            os.chdir(pwd)

            cnt_dense_train = 0

            cnt_dense_train += 1
            dense_model, dense_cnt_top, dense_cnt_mid = changeModel(
                countList, cnt_trn, trn_shp)

            for eps in range(1, EPS_MAX_RNG + 1):

                print('LAYER: ' +
                      out_layer + ' /// TEST_OBJECT: ' + str(
                          cnt_tst + 1) + ' /// HIDDEN_LYR_NB: ' +
                      str(countList) + ' /// TRAIN_NB: ' +
                      str(cnt_dense_train) + ' /// EPOCH: ' + str(eps))

                hist_fit = dense_model.fit(
                    X_train,
                    y_train,
                    nb_epoch=(1),
                    batch_size=batch_max_val,
                    shuffle=True,
                    verbose=0)

                # DECAY DO LEARNING RATE -- tinha feito uma solucao customizada pois o default do keras atualiza o decay a cada update de batch
                # ja este atualiza a cada EPS_MAX_RNG/10
                # if ((eps % (EPS_MAX_RNG/10)) == 0):
                #	K.set_value(dense_model.optimizer.lr, math.pow(1./1000., 1./EPS_MAX_RNG) * K.get_value(dense_model.optimizer.lr))

                # SALVA O MODELO NA ULTIMA EPOCA
                # if (eps == EPS_MAX_RNG):
                #	save_model_str = models_dir + '/' + 'FC_' + out_layer + '_test_batch_' + str(cnt_tst+1) + '_hidden_lyr_' + str(countList) + '_train_nb_' + str(cnt_dense_train) + '_epoch_' + str(eps)
                #	dense_model.save(save_model_str + '.h5')

                # CRONOMETRA O TEMPO QUE LEVA PARA FAZER UM FORWARD PASS DO TESTE E DA VALIDACAO
                start_time = time.time()
                proba_vec_test = dense_model.predict_proba(X_test)
                elapsed = time.time() - start_time

                proba_vec_test = np.reshape(proba_vec_test,
                                            (proba_vec_test.shape[0]))
                proba_vec = dense_model.predict_proba(X_val)
                proba_vec = np.reshape(proba_vec, (proba_vec.shape[0]))

                # CALCULO DOS DOIS MELHORES LIMIARES DE ACORDO COM A VALIDACAO
                ##############################################################
                ######## OTIMIZADO PARA A DISTANCIA E NAO ACURACIA ###########
                ##############################################################

                best_ACCR_LOW = 0
                best_THRESH_LOW = 0
                best_DISR_LOW = 999
                best_ACCR_HIGH = 0
                best_THRESH_HIGH = 0
                best_DISR_HIGH = 999
                for acc_rng in range(5, int(1e+3), 5):
                    acc_curr = acc_rng / 1e+3
                    predict_vec = np.zeros((y_val.shape[0]))
                    predict_vec[np.where(proba_vec > acc_curr)] = 1

                    TP = TN = FP = FN = 0
                    DISR = ACCR = TPR = TNR = FPR = FNR = 0
                    for i in range(y_val.shape[0]):
                        if ((y_val[i] == 1) and (predict_vec[i] == 1)):
                            TP += 1
                        elif ((y_val[i] == 0)
                                and (predict_vec[i] == 1)):
                            FP += 1
                        elif ((y_val[i] == 1)
                                and (predict_vec[i] == 0)):
                            FN += 1
                        elif ((y_val[i] == 0)
                                and (predict_vec[i] == 0)):
                            TN += 1
                    TNR = TN / (TN + FP)
                    FPR = FP / (FP + TN)
                    FNR = FN / (FN + TP)
                    TPR = TP / (TP + FN)
                    ACCR = (TN + TP) / (TN + TP + FN + FP)
                    DISR = ((1 - TPR)**2 + (FPR)**2)**(0.5)

                    if (DISR <= best_DISR_HIGH):
                        best_ACCR_HIGH = ACCR
                        best_THRESH_HIGH = acc_curr
                        best_DISR_HIGH = DISR

                    if (DISR < best_DISR_LOW):
                        best_ACCR_LOW = ACCR
                        best_THRESH_LOW = acc_curr
                        best_DISR_LOW = DISR

                # CASOS ESPECIAIS PARA OS ULTIMOS BLOCOS DE TESTE, QUE POSSUEM VIDEOS REPETIDOS SO PARA COMPLETAR O BLOCO DE 4 OBJETOS
                if (cnt_tst <= 13):
                    max_obj = 4
                elif (cnt_tst == 14):
                    max_obj = 2
                elif (cnt_tst == 15):
                    max_obj = 1
                elif (cnt_tst == 16):
                    max_obj = 1

                for cnt_obj in range(max_obj):

                    save_acc_str = 'FC_acc+thresh_' + str_objs + out_layer + '_epoch_' + str(EPS_MAX_RNG) + \
                        '_hidden_lyr_' + str(countList) + '_train_nb_' + str(
                            cnt_dense_train) + '_' + 'video' + str(VDAO_TEST_ENTRIES_OBJ_NB[cnt_tst][cnt_obj])

                    proba_vec_test_tmp = proba_vec_test[(
                        cnt_obj * 200):((cnt_obj + 1) * 200)]
                    y_test_tmp = y_test[(cnt_obj * 200):((
                        cnt_obj + 1) * 200)]
                    div_fac = 4

                    # SALVA OS RESULTADOS EM UM ARQUIVO CSV
                    # EPC - Epoca // THR - Threshold // Acc- Acuracia // DIS - distancia
                    # TNN - numero de neuronios da camada escondida do topo
                    # MNN - numero de neuronios da camada escondida do meio
                    # ELP - tempo decorrido
                    # ELPD - tempo decorrido (para cada objeto)

                    #	O ARQUIVO SEGUE A SEGUINTE ORDEM
                    #
                    # 	EPC,THR,ACC,DIS,TP,TN,FP,FN,TNN,MNN,ELP,ELPD
                    #	LINHA 1 - EPOCA 1 - Resultado para limiar de 50%
                    #	LINHA 2 - EPOCA 1 - Resultado para o melhor limiar por baixo
                    #	LINHA 3 - EPOCA 1 - Resultado para o melhor limiar por cima
                    #	LINHA 4 - EPOCA 2 - Resultado para limiar de 50%
                    #	LINHA 5 - EPOCA 2 - Resultado para o melhor limiar por baixo
                    #	LINHA 6 - EPOCA 2 - Resultado para o melhor limiar por cima
                    #	......

                    txt_file = open(save_acc_str + '.csv', 'a')
                    if eps == 1:
                        txt_file.write(
                            'EPC,THR,ACC,DIS,TP,TN,FP,FN,TNN,MNN,ELP,ELPD\n'
                        )

                    flagOnce = True
                    flagTwice = True
                    for acc_curr in [
                            0.5, best_THRESH_LOW, best_THRESH_HIGH
                    ]:
                        predict_vec = np.zeros((y_test_tmp.shape[0]))
                        predict_vec[np.where(
                            proba_vec_test_tmp > acc_curr)] = 1

                        TP = TN = FP = FN = 0
                        DISR = ACCR = TPR = TNR = FPR = FNR = 0
                        for i in range(y_test_tmp.shape[0]):
                            if ((y_test_tmp[i] == 1)
                                    and (predict_vec[i] == 1)):
                                TP += 1
                            elif ((y_test_tmp[i] == 0)
                                    and (predict_vec[i] == 1)):
                                FP += 1
                            elif ((y_test_tmp[i] == 1)
                                    and (predict_vec[i] == 0)):
                                FN += 1
                            elif ((y_test_tmp[i] == 0)
                                    and (predict_vec[i] == 0)):
                                TN += 1

                        flagTN = False
                        flagTP = False
                        try:
                            TNR = ((TN * 100.0) / (TN + FP)) / 100.0
                            FPR = ((FP * 100.0) / (FP + TN)) / 100.0
                            FNR = ((FN * 100.0) / (FN + TP)) / 100.0
                            TPR = ((TP * 100.0) / (TP + FN)) / 100.0
                        except Exception as e:
                            if (TN == 0 and FP == 0):
                                flagTN = True
                                TNR = 1
                                FPR = 0
                                FNR = ((FN * 100.0) /
                                       (FN + TP)) / 100.0
                                TPR = ((TP * 100.0) /
                                       (TP + FN)) / 100.0
                            elif (TP == 0 and FN == 0):
                                flagTP = True
                                TPR = 1
                                FNR = 0
                                TNR = ((TN * 100.0) /
                                       (TN + FP)) / 100.0
                                FPR = ((FP * 100.0) /
                                       (FP + TN)) / 100.0
                        ACCR = (((TN + TP) * 100.0) /
                                (TN + TP + FN + FP)) / 100.0
                        DISR = ((1 - TPR)**2 + (FPR)**2)**(0.5)

                        if ((acc_curr == 0.5) and (flagOnce)):
                            acc_val_str = '%d' % (eps) + ','
                            acc_val_str = acc_val_str + '%.3f' % (
                                acc_curr * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                ACCR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                DISR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                TPR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                TNR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                FPR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                FNR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                dense_cnt_top) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                dense_cnt_mid) + ','
                            acc_val_str = acc_val_str + '%f' % (
                                elapsed) + ','
                            acc_val_str = acc_val_str + '%f' % (
                                elapsed / div_fac) + '\n'
                            txt_file.write(acc_val_str)
                            flagOnce = False
                        elif (flagTwice):
                            acc_val_str = '%d' % (eps) + ','
                            acc_val_str = acc_val_str + '%.3f' % (
                                acc_curr * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                ACCR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                DISR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                TPR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                TNR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                FPR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                FNR * 100) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                dense_cnt_top) + ','
                            acc_val_str = acc_val_str + '%.2f' % (
                                dense_cnt_mid) + ','
                            acc_val_str = acc_val_str + '%f' % (
                                elapsed) + ','
                            acc_val_str = acc_val_str + '%f' % (
                                elapsed / div_fac) + '\n'
                            txt_file.write(acc_val_str)
                            flagTwice = False

                    acc_curr = best_THRESH_HIGH
                    acc_val_str = '%d' % (eps) + ','
                    acc_val_str = acc_val_str + '%.3f' % (
                        acc_curr * 100) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        ACCR * 100) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        DISR * 100) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        TPR * 100) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        TNR * 100) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        FPR * 100) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        FNR * 100) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        dense_cnt_top) + ','
                    acc_val_str = acc_val_str + '%.2f' % (
                        dense_cnt_mid) + ','
                    acc_val_str = acc_val_str + '%f' % (elapsed) + ','
                    acc_val_str = acc_val_str + '%f' % (
                        elapsed / div_fac) + '\n'
                    txt_file.write(acc_val_str)
                    txt_file.close()

                    if eps == EPS_MAX_RNG:
                        os.rename(
                            save_acc_str + '.csv',
                            mv_res_dir + '/' + save_acc_str + '.csv')

            K.clear_session()


pwd = os.getcwd()

mv_res_dir = os.path.join(pwd, 'results_ACC_FC_TEST_ARTICLE_59_VIDEOS_DIS')
if not (os.path.exists(mv_res_dir)):
    os.mkdir(mv_res_dir)

models_dir = os.path.join(pwd, 'FC_59_VIDEOS_MODELS_DIS')
if not (os.path.exists(models_dir)):
    os.mkdir(models_dir)

while (countList < 3):
    main_prog()
    countList += 1
