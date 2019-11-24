import ad
import numpy as np
import MnistLoader as Mloader
import matplotlib.pyplot as plt

def label2oneHot(label,n):
    ret=np.zeros((10,n))
    for i in range(n):
        ret[label[i],i]=1
    return ret
def unifyImages(images,n):
    ret=np.zeros((784,n))
    for i in range(n):
        ret[:,i]=images[i,:]
    return ret
def getPredResult(pred):
    return np.argmax(pred)

if __name__ == "__main__":
    images_train, label_train = Mloader.load_mnist(r"D:\PycharmProjs\AD\mnist", "train")

    N=10000
    modes=["complex","simple"]
    mode=modes[1]
    label_raw =label_train[:N]
    label_train=label2oneHot(label_train,N)
    images_train=unifyImages(images_train,N)

    if mode=="complex":
        W1=ad.Variable("W1(100*784)")
        b1=ad.Variable("b1(100*1)")
        W2=ad.Variable("W2(100*100)")
        b2=ad.Variable("b2(100*1)")
        W3=ad.Variable("w3(10*100)")
        b3=ad.Variable("b3(10*1)")
        z0=ad.Variable("input(768*1)")

        label=ad.Variable("label")

        z1=ad.matmul(W1,z0)+b1
        z2=ad.matmul(W2,z1)+b2
        z3=ad.matmul(W3,z2)+b3

        pred=ad.softmax(z3)
        J=ad.softmax_crossent(z3,label)

        executor=ad.Executor([J,pred]+ad.gradients(J,[W1,b1,W2,b2,W3,b3]))
        W1_val=np.random.random((100,784))*0.001
        b1_val=np.random.random((100,1))*0.001
        W2_val=np.random.random((100,100))*0.001
        b2_val=np.random.random((100,1))*0.001
        W3_val=np.random.random((10,100))*0.001
        b3_val=np.random.random((10,1))*0.001
        learnRate=0.01
    if mode=="simple":

        W1 = ad.Variable("W1(10*784)")
        b1 = ad.Variable("b1(10*1)")
        z0 = ad.Variable("input(768*1)")
        label = ad.Variable("label")
        z1=ad.matmul(W1,z0)+b1
        pred = ad.softmax(z1)
        J = ad.softmax_crossent(z1, label)
        executor = ad.Executor([J, pred] + ad.gradients(J, [W1, b1]))
        W1_val = np.random.random((10, 784))
        b1_val = np.random.random((10, 1))
        learnRate = 0.3


    total_num=0
    correct=0
    error=0
    acc=1
    max_epoch = 3
    for epochs in range(max_epoch):
        for i in range(N):
            z0_val=(images_train[:,i]>0).reshape((784,1))
            label_val=label_train[:,i].reshape((10,1))
            if mode=="complex":
                _,prediction,W1_g,b1_g,W2_g,b2_g,W3_g,b3_g=executor.run(feed_dict={W1:W1_val,W2:W2_val, W3:W3_val, b1:b1_val,b2:b2_val,b3:b3_val,z0:z0_val,label:label_val})
                W1_val-=learnRate*W1_g
                W2_val-=learnRate*W2_g
                W3_val-=learnRate * W3_g
                b1_val-=learnRate*b1_g
                b2_val-=learnRate * b2_g
                b3_val-=learnRate * b3_g
                #print(W3_g)
            if mode=="simple":
                _, prediction, W1_g, b1_g=executor.run(feed_dict={W1:W1_val, b1:b1_val,z0:z0_val,label:label_val})
                W1_val -= learnRate * W1_g
                b1_val -= learnRate * b1_g
            if getPredResult(prediction)==label_raw[i]:
                correct+=1
            else:
                error+=1
            total_num+=1
            acc=correct/total_num
            err=error/total_num
            print("training... epoch:",epochs,"iter",i,"acc:",acc,"err",err)

    images_test,label_test=Mloader.load_mnist(r"D:\PycharmProjs\AD\mnist", "test")
    N_test=2000
    label_test_raw = label_test[:N_test]
    label_test = label2oneHot(label_test, N_test)
    images_test = unifyImages(images_test, N_test)
    total_num = 0
    error = 0
    correct = 0
    for i in range(N_test):
        z0_val = (images_test[:, i] > 0).reshape((784, 1))
        label_val = label_test[:, i].reshape((10, 1))
        _, prediction, _1, _2 = executor.run(feed_dict={W1: W1_val, b1: b1_val, z0: z0_val, label: label_val})

        if getPredResult(prediction) == label_test_raw[i]:
            correct += 1
        else:
            error += 1
        total_num += 1
        acc = correct / total_num
        err = error / total_num
        print("testing...", "iter", i, "acc:", acc, "err", err)
    print("Function Showcasing...")
    for i in range(20):
        z0_val = (images_test[:, i] > 0).reshape((784, 1))
        label_val = label_test[:, i].reshape((10, 1))
        _, prediction, _1, _2 = executor.run(feed_dict={W1: W1_val, b1: b1_val, z0: z0_val, label: label_val})
        print("Prediction:",getPredResult(prediction))
        plt.imshow(z0_val.reshape((28,28)),cmap='Greys', interpolation='nearest')
        plt.show()

