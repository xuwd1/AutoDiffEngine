import ad
import numpy as np

def test1():
    x1=ad.Variable(name="x1")
    x2=ad.Variable(name="x2")
    x3=ad.Variable(name="x3")
    y=x1/x2-x3

    executor=ad.Executor([x1,x2,x3,y])
    print(executor.run({x1:0,x2:2,x3:3}))
    print(y)

    grad_x1,grad_x2,grad_x3=ad.gradients(y,[x1,x2,x3])
    executor_backward=ad.Executor([y,grad_x1,grad_x2,grad_x3])
    x1_val=5*np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x1_val,grad_x2_val, grad_x3_val = executor_backward.run(feed_dict={x1:x1_val,x2: x2_val, x3: x3_val})
    print(y_val, grad_x1_val,grad_x2_val, grad_x3_val)

def multiconnection():
    x1=ad.Variable("x1")
    x2=ad.Variable("x2")
    x3 = ad.Variable("x3")
    s=x1+x2
    y=(s+x3)+(5*s)
    grads=ad.gradients(y,[x1,x2,x3])
    executor=ad.Executor([y]+grads)
    feed_dict={x1:1,x2:2,x3:3}
    res=executor.run(feed_dict)
    return

def divtest():
    x1=ad.Variable("x1")
    x2=ad.Variable("x2")
    x1_val=9
    x2_val=6

    y = x1 / x2
    z = x1 / 4
    u = 5 / x2
    Ex_y=ad.Executor([y]+ad.gradients(y,[x1,x2]))
    Ex_z=ad.Executor([z]+ad.gradients(z,[x1]))
    Ex_u=ad.Executor([u]+ad.gradients(u,[x2]))
    print(Ex_y.run(feed_dict={x1:x1_val,x2:x2_val}))
    print(Ex_z.run(feed_dict={x1: x1_val}))
    print(Ex_u.run(feed_dict={x2: x2_val}))
    return

def subtest():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x1_val = 9
    x2_val = 6
    y=x1-x2
    z=x1-4
    u=8-x2
    Ex_y = ad.Executor([y] + ad.gradients(y, [x1, x2]))
    Ex_z = ad.Executor([z] + ad.gradients(z, [x1]))
    Ex_u = ad.Executor([u] + ad.gradients(u, [x2]))
    print(Ex_y.run(feed_dict={x1: x1_val, x2: x2_val}))
    print(Ex_z.run(feed_dict={x1: x1_val}))
    print(Ex_u.run(feed_dict={x2: x2_val}))
    return

def softmaxtest():
    x1=ad.Variable("x1")
    x2=ad.Variable("x2")
    d=ad.Variable("d")
    d_val=np.array([0,1,0]).reshape(3,1)
    x1_val=3*np.ones((3,1))
    x2_val=5*np.ones((3,1))
    J=ad.softmax_crossent(x1*x2,d)
    ex=ad.Executor([J]+ad.gradients(J,[x1,x2]))
    res=ex.run({x1:x1_val,x2:x2_val,d:d_val})
    pass

if __name__=="__main__":
    softmaxtest()
    #multiconnection()
    #test1()
    #divtest()
    #ubtest()