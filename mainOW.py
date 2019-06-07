import argparse
import math
import os
import statistics
import time
from datetime import datetime
from keras.utils import Sequence
#rate = 0.17899872533253378
import numpy as np
from configobj import ConfigObj
from cnnModel.data import load_data, DataGenerator
from keras.models import model_from_json
from keras.optimizers import SGD, Adam, RMSprop


from cnnModel.cnn import build_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


conf = "configuration"
config = ConfigObj(conf)
logfile = config['log']

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
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

def plot_roc(threshold, prec, rec, title, comment="", imgdir='imgdir'):
    fig=plt.figure(figsize=(10, 4))
    ax=fig.add_subplot(111)
    fig
    plt.plot(rec, prec, label = "Precision-Recall curve",color = 'red')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Pecision')
    for i, txt in enumerate(threshold):
        i = round(i, 2)
        txt = round(txt, 2)
        ax.annotate("",(rec[i], prec[i]))
    plt.grid()
    plt.savefig('{}/roc{}.pdf'.format(imgdir, comment))
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

def run(id, cv, data_params, learn_params, model=None):

    nb_instances = data_params["nb_instances"]
    nb_classes = data_params["nb_classes"]
    nb_traces = data_params["nb_traces"]
    epochs = learn_params['epochs']
    batch_size = data_params['batch_size']
    print('Building model...')

    if model is None:
        model = build_model(learn_params, nb_classes)
    print(model)

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



    start = time.time()
    # Train model on dataset
    history = model.fit_generator(generator=data_params['train_gen'],
                                  steps_per_epoch=learn_params['train_steps'],
                                  validation_data=data_params['val_gen'],
                                  validation_steps=learn_params['val_steps'],
                                  epochs=learn_params['epochs'])

    log(id, 'Training took {:.2f} sec'.format(time.time() - start))

    #    print("LEN PREDICTED {}".format(len(predicted)))
    #log_results(id, "res_{}_{}".format(dnn, id), predicted, y_test, nb_classes, resdir=config['resdir'])

    tr_loss = round(history.history['loss'][-1], 4)
    tr_acc = round(history.history['acc'][-1], 4)



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

def main(save=False, wtime=False):
    id = gen_id()
    datapath = config['datapath']
    cross_val = config.as_int('cv')
    traces = config.as_int('traces')
    dnn = config['dnn']
    seed = config.as_int('seed')
    minlen = config.as_int('minlen')

    nb_epochs = config[dnn].as_int('nb_epochs')
    batch_size = config[dnn].as_int('batch_size')
    val_split = config[dnn].as_float('val_split')
    test_split = config[dnn].as_float('test_split')
    optimizer = config[dnn]['optimizer']
    nb_layers = config[dnn].as_int('nb_layers')
    layers = [config[dnn][str(x)] for x in range(1, nb_layers + 1)]
    lr = config[dnn][optimizer].as_float('lr')
    decay = config[dnn][optimizer].as_float('decay')
    maxlen = config[dnn].as_int('maxlen')

    nb_features = 1

    log_config(id)

    start = time.time()
    print('Loading dataset {}... '.format(datapath))
    data, labels, classes = load_data(datapath,
                             minlen=minlen,
                             maxlen=maxlen,
                             traces=traces,
                             dnn_type=dnn)
    #for x in range(len(labels)):
    #    if classes[x] == "com.visiotalent.app.dump":
    #        print(x)
    #        print(labels[x])
    #        print(classes[x])


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
    #best_test_acc = None
    best_model = None

    ID = id
    indices = np.arange(nb_instances)

    App = "com.wildec.dating.meetu.dump"

    CW = 0

    for cv in range(1, cross_val + 1):
        threshold_ini = 0.0
        threshold_dif = 0.01
        threshold_iter = 99

        model = None
        np.random.shuffle(indices)
        clas = []
        index = []
        noindex = []
        for i in indices:
            if classes[i] not in clas:
                clas.append(classes[i])
            if classes[i] == App:
                index.append(i)
            else:
                noindex.append(i)


        nb_OWclas = len(index)
        print("# muestras App total",nb_OWclas)

        trainsplit = int((1-test_split-val_split) * nb_OWclas)
        testsplit = int(test_split * nb_OWclas)
        valsplit = int(val_split * nb_OWclas)

        print("# muestras App testing",trainsplit)
        clase = []

        for i in noindex:
            if classes[i] not in clase:
                clase.append(classes[i])
        print("#clases en train", len(clase))
        noindextest = int((len(noindex) - trainsplit) / 2)
        index_train = np.array(index[:trainsplit] + noindex[:trainsplit])
        index_test = np.array(index[trainsplit:(trainsplit + testsplit)] + noindex[trainsplit:(noindextest)])
        index_val = np.array(index[(trainsplit + testsplit):] + noindex[noindextest:])
        print("#traces app en test/val", testsplit)
        print("#traces noapp en test/val", int(noindextest-trainsplit))

        index_data = np.array(indices)

        # Generators
        start = time.time()
        train_gen = DataGenerator(batch_size=batch_size).generate(data, labels, index_train)
        val_gen = DataGenerator(batch_size=batch_size).generate(data, labels, index_val)
        test_gen = DataGenerator(batch_size=1).generate(data, labels, index_test)
        end = time.time()
        data_params = {'train_gen': train_gen,
                       'val_gen': val_gen,
                       # 'test_data': (x_test, y_test),
                       'nb_instances': nb_instances,
                       'nb_classes': nb_classes,
                       'nb_traces': nb_traces,
                       'epochs': nb_epochs,
                       'batch_size': batch_size}
        learn_params = {'dnn_type': dnn,
                        'epochs': nb_epochs,
                        'train_steps': index_train.shape[0] // batch_size,
                        'val_steps': index_val.shape[0] // batch_size,
                        'nb_features': nb_features,
                        'batch_size': batch_size,
                        'optimizer': optimizer,
                        'nb_layers': nb_layers,
                        'layers': layers,
                        'lr': lr,
                        'decay': decay,
                        'maxlen': maxlen}
        print("koko",index_val.shape[0] // batch_size)
        log(id, "Experiment {}: seed {}".format(cv, seed))

        tr_loss, tr_acc, model = run(id, cv, data_params, learn_params, model)

        #Predict for test dataset
        #start = time.time()

        x_test = np.take(data, axis=0, indices=index_val)
        y_test = np.take(labels, axis=0, indices=index_val)

        x_data = np.take(data, axis=0, indices=index_data)
        y_data = np.take(labels, axis=0, indices=index_data)

       # end = time.time()
        print("Took {:.2f} sec to separate the test dataset.".format(end - start))

        start = time.time()
        #predicted, test_acc, test_loss = predict(id, model, (x_test, y_test), batch_size)
        _, test_acc, test_loss = predict(id, model, test_gen, steps=len(index_test), gen=True)
        log(id, 'Test took {:.2f} sec'.format(time.time() - start))

        Y_p = model.predict_proba(x_test)
        Y_p = model.predict(x_test)
        y_p = np.argmax(Y_p, axis=1)
        Y_d = np.argmax(y_test, axis=1)
        print("Y_p", Y_p)
        print("y_p", y_p)
        print("Y_d", Y_d)
        #print(zip(model., model.predict_proba(x_test)))
        Y_pred = model.predict_proba(x_data)
        y_pred = np.argmax(Y_pred, axis=1)
        Y_data = np.argmax(y_data, axis=1)

        lbl = ["no", App]
        if lbl[0] == App:
            app = 0
            noapp = 1
        else:
            app = 1
            noapp = 0

        prec = []
        rec = []
        threshold=[]
        for t in range(0,threshold_iter):
            threshold_ini+=threshold_dif
            threshold.append(threshold_ini)
            for p in range(len(y_p)):
                if Y_p[p,app]>threshold[t]:
                    y_p[p] = app
                else:
                    y_p[p] = noapp

            recall = [0] * 100
            precision = [0] * 100
            TP = [0] * 100
            FN = [0] * 100
            FP = [0] * 100

            for i in range(len(Y_d)):
                if Y_d[i] == y_p[i]:
                    TP[Y_d[i]] += 1

                elif Y_d[i] != y_p[i]:
                    FN[Y_d[i]] += 1
                    FP[y_p[i]] += 1

            for j in range(len(lbl)):
                if (FP[j] == 0) & (TP[j]==0):
                    precision[j] = 0
                else:
                    precision[j] = TP[j] / (TP[j] + FP[j])
                if (FN[j] == 0) & (TP[j]==0):
                    recall[j] = 0
                else:
                    recall[j] = TP[j] / (TP[j] + FN[j])
                if lbl[j] != "no":
                    prec.append(precision[j])
                    rec.append(recall[j])

            #print(confusion_matrix(Y_d, y_p))
            print(threshold[t], "Label", lbl[j], "Precision",precision[j], "Recall", recall[j],  "TP", TP[j], "FP", FP[j], "FN", FN[j])
            #print(classification_report(Y_d, y_p, target_names=lbl))


        plot_roc(threshold, prec, rec, title="ROC  {}w {}tr {}bs {}epo".format(nb_classes, nb_traces, batch_size, nb_epochs),comment = "{}_{}".format(id, cv),imgdir=config['imgdir'])

        if cross_val == 1:
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
            #best_test_acc = test_acc
            best_tr_loss = tr_loss
            best_tr_acc = tr_acc

    id = ID

    if cross_val > 1:
        mean_loss = round(statistics.mean(all_test_loss), 4)
        std_loss = round(statistics.stdev(all_test_loss), 4)
        mean_acc = round(statistics.mean(all_test_acc), 4)
        std_acc = round(statistics.stdev(all_test_acc), 4)
        log(id, "CV Test loss: mean {}, std {}".format(mean_loss, std_loss))
        log(id, "CV Test accuracy: mean {}, std {}".format(mean_acc, std_acc))
        log_exp.write(";".join(list(map(lambda a: str(a), [id, nb_classes, nb_traces, dnn, nb_layers,
                                                           lr, nb_epochs, best_tr_loss, best_tr_acc,
                                                           cross_val, mean_loss, mean_acc, "+/-" + str(std_acc)]))) + "\n")

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