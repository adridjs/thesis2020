from gender_bias.svm import SVM
from matplotlib import pyplot as plt

if __name__ == '__main__':
    svm_balanced = SVM(corpus='balanced')
    svm_balanced.clustering.select_values(filter_words=True)
    svm_balanced._split()
    svm_balanced.prepare_data()
    svm_balanced.plot_disc()
    # accuracy = svm_balanced.score()
    # print(f'Balanced: {accuracy}')
    # svm_balanced.plot_discriminant(title='SVM with RBF kernel - Balanced')
    # svm_EuroParl = SVM(corpus='EuroParl')
    # accuracy = svm_EuroParl.score(train=svm_balanced.train, test=svm_balanced.test)
    # print(f'EuroParl: {accuracy}')
    # svm_EuroParl.plot_discriminant(title='SVM with RBF kernel - EuroParl')
    # plt.show()
