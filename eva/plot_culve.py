from sklearn.metrics import roc_curve,auc,average_precision_score,precision_recall_curve,confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_embedding(data):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    return data

def plot_tsne(x_test,y_test,dataset):
    if dataset == 'nsl':
        tsne = TSNE(
            n_components=3,
            perplexity=5,  #
            metric="euclidean",  #
            n_jobs=6,  #
            random_state=42,
            verbose=True,
            learning_rate=140
        )

        embedding = tsne.fit_transform(x_test, y_test)

        aim_data = plot_embedding(embedding)
        fig = plt.figure()
        ax = Axes3D(fig)
        x = aim_data[:, 0]
        y = aim_data[:, 1]
        z = aim_data[:, 2]
        ax.scatter(x,y,z,c = y_test,marker='.',cmap='Spectral')
        plt.title("T-SNE Digits")
        #plt.savefig("T-SNE_Digits_3d.png")
        elev = 40
        azim = -50
        ax.view_init(elev, azim)
        plt.show()
        print('ax.azim {}'.format(ax.azim))
        print('ax.elev {}'.format(ax.elev))


    elif dataset == 'cicids':

        tsne = TSNE(
            n_components=3,
            perplexity=40,  #
            metric="euclidean",  #
            n_jobs=6,  #
            random_state=42,
            verbose=True,
        )

        x_test = x_test[7000:8100]
        y_test = y_test[7000:8100]
        embedding = tsne.fit_transform(x_test, y_test)
        aim_data = plot_embedding(embedding)
        fig = plt.figure()
        ax = Axes3D(fig)
        x = aim_data[:, 0]
        y = aim_data[:, 1]
        z = aim_data[:, 2]
        ax.scatter(x,y,z,c = y_test,marker='.',cmap='tab10')
        plt.title("T-SNE Digits")
        #plt.savefig("T-SNE_Digits_3d.png")
        plt.show()
    else:
        tsne = TSNE(
            n_components=3,
            perplexity=10,  #
            metric="euclidean",  #
            n_jobs=6,  #
            random_state=42,
            verbose=True,
        )

        #data, x_test, y_test = main_process(100)
        #x_test = x_test[69500:70500]
        #y_test = y_test[69500:70500]
        embedding = tsne.fit_transform(x_test, y_test)
        aim_data = plot_embedding(embedding)
        fig = plt.figure()
        ax = Axes3D(fig)
        x = aim_data[:, 0]
        y = aim_data[:, 1]
        z = aim_data[:, 2]
        ax.scatter(x,y,z,c = y_test,marker='.',cmap='tab20b')
        plt.title("T-SNE Digits")
        #plt.savefig("T-SNE_Digits_3d.png")
        plt.show()

def plot_roc_auc(y_test, y_test_scores, name):
    '''

    :param y_test: train data
    :param y_test_scores: prob
    :param name: conbined name
    :return: figure
    '''

    fpr, tpr, threshod = roc_curve(y_test, y_test_scores)
    fpr = fpr.reshape(fpr.shape[0], 1)
    tpr = tpr.reshape(tpr.shape[0], 1)
    data_save = np.concatenate((fpr, tpr),axis=1)
    #tips = sns.load_dataset("tips")
    #print(tips)
    #sns.lmplot(data=data_save, x="total_bill", y="tip", col="time", hue="smoker")
    #plt.show()
    writerCSV = pd.DataFrame(data=data_save)
    # array = np.array(Loss_list)
    writerCSV.to_csv('./csv_data/%s_tpr.csv'%name,encoding='utf-8')
    print(fpr.shape)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, 'darkorange', label='ROC (area = {0:.4f})'.format(roc_auc), lw=2)
    print('auc:',roc_auc)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('%s ROC Curve' % name)
    plt.legend(loc="lower right")
    plt.show()

def plot_ROC(y_test, recon_error_test,name):
    fpr, tpr, thresholds = roc_curve(y_test, recon_error_test)
    roc_auc = auc(fpr, tpr)

    plt.title('Normal vs %s'%name)
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC',dpi=1200)
    plt.show()

def plt_loss2(x1,y1,name):
    fig = plt.figure(figsize=(7, 5))
    #pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')`
    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    p2 = pl.plot(x1, y1,'r-', label = u'%s_Net'%name)
    pl.legend()
    #显示图例
    #p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
    #pl.legend()
    pl.xlabel(u'iters')
    pl.ylabel(u'loss')
    plt.title('Compare loss for different models in training')


def plot_PRC(y_test, recon_error_test):
    average_precision = average_precision_score(y_test, recon_error_test)

    precision,recall,thresholds_prc = precision_recall_curve(y_test, recon_error_test)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(
              average_precision))
    plt.savefig('PRC',dpi=1200)
    plt.show()

def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def plt_loss(num_epochs,Loss_list):
    """
    :param num_epochs:训练轮数
    :param Loss_list: 损失函数的list
    :return:
    """
    x2 = range(0, num_epochs)
    # y1 = Accuracy_list
    y2 = Loss_list
    #plt.subplot(2, 1, 1)
    plt.plot(x2, y2, 'o-')
    # plt.title('Test accuracy vs. epoches')
    # plt.ylabel('Test accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '.-')
    plt.xlabel('loss vs. epoches')
    plt.ylabel(' loss')
    plt.show()

def plot_loss_new(num_epochs,Loss_list):
    x = range(0, num_epochs)
    y = Loss_list
    pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')

    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    # p2 = pl.plot(x1, y1,'r-', label = u'RCSCA_Net')
    pl.legend()
    pl.xlabel(u'iters')
    pl.ylabel(u'loss')
    plt.title(' loss for the model')
    plt.show()

def plot_box(losses_normal,losses_malware,name):
    plt.figure(figsize=(8,5))
    plt.title('%s of boxplot'%name, fontsize=20)
    labels = 'normal', 'malware'
    plt.boxplot([losses_normal, losses_malware], labels=labels)
    plt.show()

if __name__ == '__main__':
    classes = ['A', 'B', 'C', 'D', 'E', 'F']

    random_numbers = np.random.randint(6, size=50)  # 6个类别，随机生成50个样本
    y_true = random_numbers.copy()  # 样本实际标签
    random_numbers[:10] = np.random.randint(6, size=10)  # 将前10个样本的值进行随机更改
    y_pred = random_numbers  # 样本预测标签

    # 获取混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, 'confusion_matrix.png',classes ,title='confusion matrix')