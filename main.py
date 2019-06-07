import argparse
import math
import os
import statistics
import time
from datetime import datetime
from keras.utils import Sequence
#rate = 0.17899872533253378
import numpy as np
import pandas as pd
from configobj import ConfigObj
from cnnModel.data import load_data, DataGenerator
from keras.models import model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

from cnnModel.cnn import build_model
from sklearn.metrics import confusion_matrix, classification_report


conf = "configuration"
config = ConfigObj(conf)
logfile = config['log']

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
#matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt


def numpy_printopts(float_precision=6):
    float_formatter = lambda x: "%.{}f".format(float_precision) % x
    np.set_printoptions(formatter={'float_kind': float_formatter})  # precision


def curtime():
    return datetime.utcnow().strftime('%d.%m %H:%M:%S') #.%f')[:-3]


def log(id, s, dnn=None):
    print("> {}".format(s))
    if dnn is not None:
        l = open(dnn + "_log.out", "a")
    else:
        l = open(logfile, "a")
    l.write("ID{} {}>\t{}\n".format(id, curtime(), s))
    l.close()


def log_config(id):
    l = open("log_configs.out", "a")
    l.write("\nID{} {}\n".format(id, datetime.utcnow().strftime('%d.%m')))
    l.writelines(open(conf, 'r').readlines())
    l.close()


def gen_id():
    return datetime.utcnow().strftime('%d%m_%H%M%S')


def plot_acc(acc, title, val_acc=None, comment="", imgdir='imgdir'):
    plt.figure(figsize=(10, 4))
    plt.ylim(0, 1)
    plt.plot(acc, label="Training", color='red')
    if val_acc is not None:
        plt.plot(val_acc, label="Validation", color='blue')
    plt.title(title, y=1.08)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode='expand', borderaxespad=0.)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig('{}/acc{}.pdf'.format(imgdir, comment))
    plt.close()


def plot_loss(loss, title, val_loss=None, comment="", imgdir='imgdir'):
    plt.figure(figsize=(10, 4))
    plt.plot(loss, label="Training", color='purple')
    if val_loss is not None:
        plt.ylim(0, 5)
        plt.plot(val_loss, label="Validation", color='green')
    plt.title(title, y=1.08)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode='expand', borderaxespad=0.)
    plt.yticks(np.arange(0, 5, 0.5))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig('{}/loss{}.pdf'.format(imgdir, comment))
    plt.close()


def entropy(probs):
    e = 0.0
    for prob in probs:
        if prob == 0.0:
            continue
        e += prob * math.log(prob) #ln
    return -e


def log_results(id, fname, predicted, nb_classes, dnn=None, labels=None, resdir='resdir'):
    r = open("{}/{}.csv".format(resdir, fname), "w")

    r.write("correct;label;predicted;predicted_prob;entropy")

    for cl in range(0, nb_classes):
        r.write(";prob_{}".format(cl))

    r.write("\n")

    class_result = np.argmax(predicted, axis=-1)

    acc = 0.0
    num = len(predicted)
    for res in range(0, num):
        predicted_label = int(class_result[res])
        prob = predicted[res][predicted_label]
        ent = entropy(predicted[res])

        if labels is not None:
            label = int(np.argmax(labels[res], axis=-1))
            correct = int(label == predicted_label)
            acc += correct
        else:
            label = "-"
            correct = "-"
        r.write("{};{};{};{:.4f};{:.4f}".format(correct, label, predicted_label, prob, ent))
        for cl in range(0, nb_classes):
            r.write(";{:.4f}".format(predicted[res][cl]))
        r.write("\n")
    r.close()
    if labels is not None:
        acc /= num
        log(id, "Accuracy:\t{}".format(acc), dnn)

    log(id, "PredictionsB saved to {}".format(fname), dnn)


def predict(id, model, data, batch_size=1, steps=0, gen=False):
    if gen:
        score = model.evaluate_generator(data, steps)

        predicted = model.predict_generator(data, steps)
    else:
        (x, y) = data
        score = model.evaluate(x, y, batch_size=batch_size, verbose=1)
        predicted = model.predict(x)

    test_loss = round(score[0], 4)
    test_acc = round(score[1], 4)
    log(id, "Test loss(entropy):\t{}".format(test_loss))
    log(id, "Test accuracy:\t{}".format(test_acc))

    return predicted, test_acc, test_loss


def run(title, id, fold, data_params, learn_params, model=None):

    nb_instances = data_params["nb_instances"]
    nb_classes = data_params["nb_classes"]
    nb_traces = data_params["nb_traces"]
    epochs = learn_params['epochs']
    batch_size = data_params['batch_size']
    print('Building model...')

    if model is None:
        model = build_model(learn_params, nb_classes)

    metrics = ['accuracy']

    optimizer = None
    if learn_params['optimizer'] == "sgd":
        optimizer = SGD(lr=learn_params['lr'],
                        decay=learn_params['decay'],
                        momentum=0.9,
                        nesterov=True)
    elif learn_params['optimizer'] == "adam":
        optimizer = Adam(lr=learn_params['lr'],
                         decay=learn_params['decay'])
    else:  # elif learn_params['optimizer'] == "rmsprop":
        optimizer = RMSprop(lr=learn_params['lr'],
                            decay=learn_params['decay'])

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    print(model.summary())

    epochs=learn_params['epochs']

    start = time.time()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    checkpointer = ModelCheckpoint(filepath='lstm_weights.hdf5',
                                   verbose=0,
                                   save_best_only=True)
    # Train model on dataset
    history = model.fit_generator(generator=data_params['train_gen'],
                                  steps_per_epoch=learn_params['train_steps'],
                                  validation_data=data_params['val_gen'],
                                  validation_steps=learn_params['val_steps'],
                                  callbacks=[early_stopping, checkpointer],
                                  epochs=learn_params['epochs'])

    log(id, 'Training took {:.2f} sec'.format(time.time() - start))

    tr_loss = round(history.history['loss'][-1], 4)
    tr_acc = round(history.history['acc'][-1], 4)


    plot_loss(history.history['loss'],
             title="",
              val_loss=history.history['val_loss'],
              comment="{}{}fold" .format(title, fold),
              imgdir =config['imgdir'])
    plot_acc(history.history['acc'],
             title="",
             val_acc=history.history['val_acc'],
              comment="{}{}fold" .format(title, fold),
             imgdir=config['imgdir'])
    return tr_loss, tr_acc, model


def parse_model_name(model_path):
    name = os.path.basename(model_path)
    return name.split("_")[0] + "_" + name.split("_")[1], name.split("_")[2]


def load_model(model_path):
    # load json and create model
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path + ".h5")
    return loaded_model

def create_folds(indices,classes, nb_traces, nb_folds):

    clase = {}
    clas = []
    kfold1 = []
    kfold2 = []
    kfold3 = []
    kfold4 = []
    kfold5 = []
    kfold6 = []
    kfold7 = []
    kfold8 = []
    kfold9 = []
    kfold10 = []
    for i in indices:

        if classes[i] not in clase:
            clas.append(classes[i])
            clase[classes[i]] = 1
        if (clase[classes[i]] <= (nb_traces/nb_folds)) & (nb_folds>=1):
            count = clase[classes[i]]
            kfold1.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 2*(nb_traces/nb_folds)) & (nb_folds>=2):
            count = clase[classes[i]]
            kfold2.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 3*(nb_traces/nb_folds)) & (nb_folds>=3):
            count = clase[classes[i]]
            kfold3.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 4*(nb_traces/nb_folds)) & (nb_folds>=4):
            count = clase[classes[i]]
            kfold4.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 5*(nb_traces/nb_folds)) & (nb_folds>=5):
            count = clase[classes[i]]
            kfold5.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 6*(nb_traces/nb_folds)) & (nb_folds>=6):
            count = clase[classes[i]]
            kfold6.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 7*(nb_traces/nb_folds)) & (nb_folds>=7):
            count = clase[classes[i]]
            kfold7.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 8*(nb_traces/nb_folds)) & (nb_folds>=8):
            count = clase[classes[i]]
            kfold8.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 9*(nb_traces/nb_folds)) & (nb_folds>=9):
            count = clase[classes[i]]
            kfold9.append(i)
            clase[classes[i]] = count + 1
        elif (clase[classes[i]] <= 10*(nb_traces/nb_folds)) & (nb_folds>=10):
            count = clase[classes[i]]
            kfold10.append(i)
            clase[classes[i]] = count + 1
        kfold = [kfold1, kfold2, kfold3, kfold4, kfold5, kfold6, kfold7, kfold8, kfold9, kfold10]

    return kfold, clas


def circleData(fold, nb_folds, kfold):
    ind_train=[]
    for f in range(nb_folds):
        if f == fold:
            ind_val = kfold[f]
            if fold == (nb_folds-1):
                ind_test = kfold[0]
        elif f - 1 == fold:
            ind_test = kfold[f]
        else:
            ind_train = ind_train + kfold[f]
    print(len(ind_val), len(ind_test), len(ind_train))

    return ind_train, ind_test, ind_val


def main(save=False, wtime=False):
    id = gen_id()
    datapath = config['datapath']
    traces = config.as_int('traces')
    dnn = config['dnn']
    minlen = config.as_int('minlen')

    nb_epochs = config[dnn].as_int('nb_epochs')
    title = config['title']
    batch_size = config[dnn].as_int('batch_size')
    val_split = config[dnn].as_float('val_split')
    test_split = config[dnn].as_float('test_split')
    optimizer = config[dnn]['optimizer']
    nb_layers = config[dnn].as_int('nb_layers')
    layers = [config[dnn][str(x)] for x in range(1, nb_layers + 1)]
    lr = config[dnn][optimizer].as_float('lr')
    decay = config[dnn][optimizer].as_float('decay')
    maxlen = config[dnn].as_int('maxlen')
    nb_folds = config[dnn].as_int('nb_folds')
    nb_features = 1

    log_config(id)

    start = time.time()
    print('Loading dataset {}... '.format(datapath))
    data, labels, classes = load_data(datapath,
                             minlen=minlen,
                             maxlen=maxlen,
                             traces=traces,
                             dnn_type=dnn)

    end = time.time()
    print("Took {:.2f} sec to load.".format(end - start))

    nb_instances = data.shape[0]
    nb_classes = labels.shape[1]
    nb_traces = int(nb_instances / nb_classes)

    log(id, 'Loaded dataset {} instances for {} classes: '
            '{} traces per class'.format(nb_instances,
                                        nb_classes,
                                        nb_traces,))

    # CROSS-VALIDATION
    log_exp_name = "experiments.csv"
    if os.path.isfile(log_exp_name):
        log_exp = open(log_exp_name, "a")
    else:
        log_exp = open(log_exp_name, "a")
        log_exp.write(";".join(["ID", "w", "tr", "tr length", "DNN", "N layers", "lr",
                                "epochs", "tr loss", "tr acc", "cv", "test loss", "test acc", "std acc"]) + "\n")

    all_test_acc = []
    all_test_loss = []

    best_tr_loss = None
    best_tr_acc = None
    best_test_loss = 10

    ID = id
    indices = np.arange(nb_instances)
    #np.random.shuffle(indices)

    start = time.time()
    kfold, clas = create_folds(indices, classes, nb_traces, nb_folds)

    end = time.time()
    print("Took {:.2f} sec to split the dataset into 10 folds.".format((end - start)))
    lis = []
    meanPR = [0] * 75
    meanRE = [0] * 75
    PR=[0] * 75
    RE=[0] * 75

    for fold in range(nb_folds):

        model = None

        # Split the dataset into training, validation and test sets
        ind_train, ind_test, ind_val = circleData(fold,nb_folds, kfold)
        start = time.time()
        end = time.time()
        print("Took {:.2f} sec to split the dataset into training, validation and test sets".format(end - start))

        # Generators
        start = time.time()
        train_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_train)
        val_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_val)
        test_gen = DataGenerator(batch_size=1).generate(data, labels, ind_test)
        end = time.time()
        print("Took {:.2f} sec to make the generators.".format(end - start))
        data_params = {'train_gen': train_gen,
                       'val_gen': val_gen,
                        'nb_instances': nb_instances,
                        'nb_classes': nb_classes,
                        'nb_traces': nb_traces,
                        'epochs': nb_epochs,
                        'batch_size': batch_size}
        learn_params = {'dnn_type': dnn,
                        'epochs': nb_epochs,
                        'train_steps': len(ind_train) // batch_size,
                        'val_steps': len(ind_val) // batch_size,
                        'nb_features': nb_features,
                        'batch_size': batch_size,
                        'optimizer': optimizer,
                        'nb_layers': nb_layers,
                        'layers': layers,
                        'lr': lr,
                        'decay': decay,
                        'maxlen': maxlen}

        log(id, "Experiment {}:".format(nb_folds))

        tr_loss, tr_acc, model = run(title, id, fold, data_params, learn_params, model)

        #Predict for test dataset
        start = time.time()
        x_test = np.take(data, axis=0, indices=ind_val)
        y_test = np.take(labels, axis=0, indices=ind_val)
        end = time.time()
        print("Took {:.2f} sec to separate the test dataset.".format(end - start))

        start = time.time()
        _, test_acc, test_loss = predict(id, model, test_gen, steps=len(ind_test), gen=True)

        log(id, 'Test took {:.2f} sec'.format(time.time() - start))

        Y_p = model.predict(x_test)
        y_p = np.argmax(Y_p, axis=1)
        Y_d = np.argmax(y_test, axis=1)


        recall=[0] * 1000
        prec=[0] * 1000
        TP = [0] * 1000
        FN = [0] * 1000
        FP = [0] * 1000

        lblt=[]


        for y in range(len(Y_d)):
            if (clas[Y_d[y]]  not in lblt):
                lblt.append(clas[Y_d[y]])
            if (clas[y_p[y]] not in lblt):
                lblt.append(clas[y_p[y]])
            if Y_d[y] == y_p[y]:
                TP[Y_d[y]] += 1
            else:
                FN[Y_d[y]] += 1
                FP[y_p[y]] += 1

        for i in range(len(Y_d)):
            if (FP[Y_d[i]] == 0) & (TP[Y_d[i]] == 0):
                 prec[Y_d[i]] = 0
            else:
                prec[Y_d[i]] = TP[Y_d[i]] / (TP[Y_d[i]] + FP[Y_d[i]])
            if (FN[Y_d[i]] == 0) & (TP[Y_d[i]] == 0):
                recall[Y_d[i]] = 0
            else:
                recall[Y_d[i]] = TP[Y_d[i]] / (TP[Y_d[i]] + FN[Y_d[i]])
            if clas[Y_d[i]] not in lis:
                lis.append(clas[Y_d[i]])

                #print("d:", Y_d[i], clas[Y_d[i]], "TP", TP[Y_d[i]], "FP", FP[Y_d[i]], "FN", FN[Y_d[i]], prec[Y_d[i]], recall[Y_d[i]])

        c = 0
        llista=[]
        for l in lis:
            for i in range(len(Y_d)):
                if (l == clas[Y_d[i]]) & (clas[Y_d[i]] not in llista):
                    llista.append(clas[Y_d[i]])
                    PR[c] = prec[Y_d[i]] + PR[c]
                    RE[c] = recall[Y_d[i]] + RE[c]
                    #print(l, "TP", TP[Y_d[i]], "FP", FP[Y_d[i]], "FN", FN[Y_d[i]], prec[Y_d[i]], recall[Y_d[i]])
            c += 1



        print(classification_report(Y_d, y_p, target_names=clas))
        if nb_folds == 1:
            best_model = model
            log_exp.write(";".join(list(map(lambda a: str(a), [id, nb_classes, nb_traces, dnn, nb_layers,
                                                                   lr, nb_epochs, tr_loss, tr_acc,
                                                                   "x", test_loss, test_acc, "x"]))) + "\n")
            break

        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)

        if test_loss < best_test_loss:
            best_model = model
            best_test_loss = test_loss
            best_test_acc = test_acc
            best_tr_loss = tr_loss
            best_tr_acc = tr_acc

    for l in range(len(lis)):
        meanPR[l] = (PR[l] / nb_folds)
        meanRE[l] = RE[l] / nb_folds
        print(lis[l], "{0:.3f}".format(meanPR[l]), "{0:.3f}".format(meanRE[l]))
    id = ID

    if nb_folds > 1:
            mean_loss = round(statistics.mean(all_test_loss), 4)
            std_loss = round(statistics.stdev(all_test_loss), 4)
            mean_acc = round(statistics.mean(all_test_acc), 4)
            std_acc = round(statistics.stdev(all_test_acc), 4)
            log(id, "CV Test loss: mean {}, std {}".format(mean_loss, std_loss))
            log(id, "CV Test accuracy: mean {}, std {}".format(mean_acc, std_acc))
            log_exp.write(";".join(list(map(lambda a: str(a), [id, nb_classes, nb_traces, dnn, nb_layers,
                                                               lr, nb_epochs, best_tr_loss, best_tr_acc,
                                                                nb_folds, mean_loss, mean_acc, "+/-" + str(std_acc)]))) + "\n")

#Main Run Thread
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a deep neural network CNN')

    parser.add_argument('--save', '-s',
                        action="store_true",
                        help='save the trained model (for cv: the best one)')
    parser.add_argument('--wtime', '-wt',
                        action="store_true",
                        help='time experiment: test time datasets')
    parser.add_argument('--eval', '-e',
                        action="store_true",
                        help='test the model')

    args = parser.parse_args()

    if not args.eval:
        main(args.save, args.wtime)
