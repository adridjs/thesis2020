from gender_bias.svm import SVM
from matplotlib import pyplot as plt

if __name__ == '__main__':
    svm_balanced = SVM(corpus='balanced')
    svm_balanced.clustering.select_values(filter_words=True)
    svm_balanced._split()
    svm_balanced.prepare_data()
    svm_balanced.plot_disc()
    accuracy = svm_balanced.score()
    print(f'Balanced: {accuracy}')
    svm_EuroParl = SVM(corpus='EuroParl')
    svm_EuroParl.clustering.select_values(filter_words=True)
    f = open('data/svm_words.txt', 'w+')
    svm_words = list(svm_balanced.train.keys()) + list(svm_balanced.test.keys())
    f.writelines('\n'.join(svm_words))
    svm_EuroParl._split(train=svm_balanced.train, test=svm_balanced.test)
    svm_EuroParl.prepare_data()
    svm_balanced.plot_disc()
    accuracy = svm_EuroParl.score(train=svm_balanced.train, test=svm_balanced.test)
    print(f'EuroParl: {accuracy}')
