import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt; import matplotlib.colors as clr; from matplotlib import cm
from numpy import genfromtxt

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

parser = argparse.ArgumentParser()
parser.add_argument('--init',default='random',type=str,help='initial state')
parser.add_argument('--model',default='iter7000.ckpt',type=str,help='model')
parser.add_argument('--dim',default=60,type=int,help='board height & width')
parser.add_argument('--n',default=8,type=int,help='initial number of filters')
parser.add_argument('--r',default=6,type=int,help='look radius')
parser.add_argument('--lr',default=0.0001,type=float,help='learning rate')
parser.add_argument('--gamma',default=0.1,type=float,help='discount rate')
parser.add_argument('--epsilon',default=0.1,type=float,help='probability of random action')
parser.add_argument('--epochs',default=2000000,type=int,help='training iterations')
parser.add_argument('--test_steps',default=10,type=int,help='test steps')
parser.add_argument('--train_model',default=True,action='store_false',help='train new model')
args = parser.parse_args()


def init(args):

    #NOTE: only dim = {3,6,9,12,15,21,...} supported for now
    #      uses position of nodes to randomly initializes actions
    #      -ex. dim = 6 (a=action, n=node, -1=fill; nodes are also represented by -1 when implemented, they are denoted n here for clarity):
    #      [[-1, a,-1,-1,a,-1]
    #       [ a, n, a, a,n, a]
    #       [-1, a,-1,-1,a,-1]
    #       [-1, a,-1,-1,a,-1]
    #       [ a, n, a, a,n, a]
    #       [-1, a,-1,-1,a,-1]]

    if args.init=='random':
        state = np.random.rand(args.dim,args.dim)
        for i in range(1,state.shape[0],3):
            for j in range(1,state.shape[1],3):
                state[i+1,j] = np.random.randint(0,high=2,dtype=int)
                state[i-1,j] = np.random.randint(0,high=2,dtype=int)
                state[i,j+1] = np.random.randint(0,high=2,dtype=int)
                state[i,j-1] = np.random.randint(0,high=2,dtype=int)
        return state

    if args.init=='lr':
        state = np.full((args.dim,args.dim),-1,dtype=int)
        for i in range(1,state.shape[0],3):
            for j in range(1,state.shape[1],3):
                if j<int(state.shape[1]/2):
                    state[i+1,j] = 0
                    state[i-1,j] = 0
                    state[i,j+1] = 0
                    state[i,j-1] = 0
                else:
                    state[i+1,j] = 1
                    state[i-1,j] = 1
                    state[i,j+1] = 1
                    state[i,j-1] = 1
        return state



def payoff(x):
    right = np.roll(x,-1,axis=-1)
    left = np.roll(x,1,axis=-1)
    down = np.roll(x,-1,axis=0)
    up = np.roll(x,1,axis=0)

    payoff_matrix = np.zeros(x.shape,dtype=np.float64)
    p = np.zeros(int((x.shape[0]/3)**2))
    index = 0
    cc_payoff = 5; dd_payoff = 1; cd_payoff = 0; dc_payoff=10

    for i in range(1,x.shape[0],3):
        for j in range(1,x.shape[1],3):
            # down neighbor score
            cc = np.logical_and(x[i+1,j],down[i+1,j]).astype(np.float64)
            dd = np.logical_and(np.logical_not(x[i+1,j]),np.logical_not(down[i+1,j])).astype(np.float64)
            cd = np.logical_and(x[i+1,j],np.logical_not(down[i+1,j])).astype(np.float64)
            dc = np.logical_and(np.logical_not(x[i+1,j]),down[i+1,j]).astype(np.float64)

            # up neighbor score
            cc += np.logical_and(x[i-1,j],up[i-1,j]).astype(np.float64)
            dd += np.logical_and(np.logical_not(x[i-1,j]),np.logical_not(up[i-1,j])).astype(np.float64)
            cd += np.logical_and(x[i-1,j],np.logical_not(up[i-1,j])).astype(np.float64)
            dc += np.logical_and(np.logical_not(x[i-1,j]),up[i-1,j]).astype(np.float64)

            # left neighbor score
            cc += np.logical_and(x[i,j-1],left[i,j-1]).astype(np.float64)
            dd += np.logical_and(np.logical_not(x[i,j-1]),np.logical_not(left[i,j-1])).astype(np.float64)
            cd += np.logical_and(x[i,j-1],np.logical_not(left[i,j-1])).astype(np.float64)
            dc += np.logical_and(np.logical_not(x[i,j-1]),left[i,j-1]).astype(np.float64)
            
            # right neighbor score
            cc += np.logical_and(x[i,j+1],right[i,j+1]).astype(np.float64)
            dd += np.logical_and(np.logical_not(x[i,j+1]),np.logical_not(right[i,j+1])).astype(np.float64)
            cd += np.logical_and(x[i,j+1],np.logical_not(right[i,j+1])).astype(np.float64)
            dc += np.logical_and(np.logical_not(x[i,j+1]),right[i,j+1]).astype(np.float64)

            payoff_matrix[i,j] = (cc*cc_payoff)+(dd*dd_payoff)+(cd*cd_payoff)+(dc*dc_payoff)
            p[index] = (cc*cc_payoff)+(dd*dd_payoff)+(cd*cd_payoff)+(dc*dc_payoff)
            index+=1

    return p

def recenter(state,pos,r):
    
    center = (int(state.shape[0]/2),int(state.shape[1]/2))
    v = np.roll(state,((center[0]-pos[0]),(center[1]-pos[1])),axis=(0,1))
    return v[int(v.shape[0]/2)-r:int(v.shape[0]/2)+r,int(v.shape[1]/2)-r:int(v.shape[1]/2)+r]


def batch(dim,r,state):
    #       converts state into network input
    batch = np.zeros((int((dim/3)**2),2*r,2*r,1)) #add extra channel for convolution
    index = 0
    for i in range(1,state.shape[0],3):
        for j in range(1,state.shape[1],3):
            batch[index] = np.expand_dims(recenter(state,(i,j),r),axis=4)
            index+=1

    return batch


def next_state(args,x,training):
    next_state = np.random.rand(int(x.shape[0]**(1/2))*3,int(x.shape[0]**(1/2))*3)
    bit_repr = np.zeros(x.shape[0],dtype=int)
    for i in range(0,x.shape[0]):
        bit_repr[i] = int(np.base_repr(x[i]))

        #padding
        if len(str(bit_repr[i]))==3:
            bit_repr[i] = int(str(bit_repr[i])+'0')

        elif len(str(bit_repr[i]))==2:
            bit_repr[i] = int(str(bit_repr[i])+'00')

        elif len(str(bit_repr[i]))==1:
            bit_repr[i] = int(str(bit_repr[i])+'000')

    index = 0
    for i in range(1,next_state.shape[0],3):
        for j in range(1,next_state.shape[1],3):

            p_random_action = np.random.uniform()
            if (p_random_action > args.epsilon) and training:
                next_state[i-1,j] = np.random.randint(0,high=2,dtype=int)
                next_state[i+1,j] = np.random.randint(0,high=2,dtype=int)
                next_state[i,j+1] = np.random.randint(0,high=2,dtype=int)
                next_state[i,j-1] = np.random.randint(0,high=2,dtype=int)

            if bit_repr[index]==0:
                next_state[i-1,j] = 0
                next_state[i+1,j] = 0
                next_state[i,j+1] = 0
                next_state[i,j-1] = 0
                index+=1
            else:
                left = int(str(bit_repr[index])[0])
                right = int(str(bit_repr[index])[1])
                up = int(str(bit_repr[index])[2])
                down = int(str(bit_repr[index])[3])
                next_state[i-1,j] = left 
                next_state[i+1,j] = right
                next_state[i,j+1] = up
                next_state[i,j-1] = down
                index+=1

    return next_state

def qnet(args,x,reuse):

    with tf.variable_scope("qnet"):
        q = tf.layers.conv2d(inputs=x,filters=args.n,kernel_size=args.r,strides=(1,1),activation=tf.nn.elu,padding='same');print(q)
        q = tf.layers.conv2d(inputs=q,filters=4,kernel_size=args.r,strides=(1,1),activation=tf.nn.elu,padding='same');print(q)
        q = tf.layers.conv2d(inputs=q,filters=2,kernel_size=args.r,strides=(1,1),activation=tf.nn.elu,padding='same');print(q)
        q = tf.layers.conv2d(inputs=q,filters=1,kernel_size=int(args.r/2),strides=(1,1),activation=tf.nn.elu,padding='same');print(q)

        q = tf.layers.flatten(q);print(q)
        q = tf.layers.dense(inputs=q,units=500,activation=tf.nn.elu);print(q)
        q = tf.layers.dense(inputs=q,units=250,activation=tf.nn.elu);print(q)
        q = tf.layers.dense(inputs=q,units=16,activation=None);print(q)

    return q

def target_net(args,x,reuse):

    with tf.variable_scope("target_net"):
        q = tf.layers.conv2d(inputs=x,filters=args.n,kernel_size=args.r,strides=(1,1),activation=tf.nn.elu,padding='same');print(q)
        q = tf.layers.conv2d(inputs=q,filters=4,kernel_size=args.r,strides=(1,1),activation=tf.nn.elu,padding='same');print(q)
        q = tf.layers.conv2d(inputs=q,filters=2,kernel_size=args.r,strides=(1,1),activation=tf.nn.elu,padding='same');print(q)
        q = tf.layers.conv2d(inputs=q,filters=1,kernel_size=int(args.r/2),strides=(1,1),activation=tf.nn.elu,padding='same');print(q)

        q = tf.layers.flatten(q);print(q)
        q = tf.layers.dense(inputs=q,units=500,activation=tf.nn.elu);print(q)
        q = tf.layers.dense(inputs=q,units=250,activation=tf.nn.elu);print(q)
        q = tf.layers.dense(inputs=q,units=16,activation=None);print(q)

    return q

def visualise(state,step):

    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if (state[i,j]==0.) or (state[i,j]==1.):
                pass
            else:
                state[i,j]=-1

    for i in range(1,state.shape[0],3):
        for j in range(1,state.shape[1],3):
            state[i,j] = 2
    colors = ['blue','red','green','purple']
    cmap = clr.ListedColormap(colors)

    plt.matshow(state,cmap=cmap)#,norm=norm)
    plt.savefig('save/basic_vis'+step+'.png')
    plt.show()
    plt.close()

    pass

def learning_dynamics(args,m,saver):
    test_ = init(args)
    visualise(test_,'initial_frame')
    for i in range(0,m,1000):
        print(i)
        with tf.Session() as sess:
            saver.restore(sess,'/save_model/max'+str(i)+'.ckpt')
            batch_test_ = batch(args.dim,args.r,test_)
            out_test_ = sess.run([qnet_choice],feed_dict={state:batch_test_})[0]
            test_t1_ =  next_state(args,out_test_,False)
            visualise(test_t1_,str(int(i/1000)))

def save_cooperation(args,saver):
    sample_means = []
    sample_sds = []
    action_network_means = []
    target_network_means = []
    action_network_sds = []
    target_network_sds = []
    for i in range(0,100000,1000):
        sample_cooperation = []
        action_network_value_ = []
        target_network_value_ = []
        with tf.Session() as sess:
            saver.restore(sess,'final/save_model/iter'+str(i)+'.ckpt')
            for i in range(0,100):
                test_ = init(args)
                batch_ = batch(args.dim,args.r,test_)
                out_  = sess.run([qnet_choice],feed_dict={state:batch_})[0]
                action_value_ = sess.run([qnet_value],feed_dict={state:batch_})[0]
                target_value_ = sess.run([target_net_value],feed_dict={state:batch_})[0]
                action_network_value_.append(np.mean((action_value_)))
                target_network_value_.append(np.mean((target_value_)))
                next_ = next_state(args,out_,False)
                unique, counts  = np.unique(next_, return_counts=True)
                d = dict(zip(unique,counts))
                try: 
                    sample_cooperation.append((float(d[1.0])/1600.)) #1600 actions per sample, count how many of them are cooperative
                except KeyError:
                    sample_cooperation.append(0.)
            sample_means.append(np.mean(np.asarray(sample_cooperation)))
            sample_sds.append(np.std(np.asarray(sample_cooperation)))
            action_network_means.append(np.mean(np.asarray(action_network_value_).flatten()))
            action_network_sds.append(np.std(np.asarray(action_network_value_).flatten()))
            target_network_means.append(np.mean(np.asarray(target_network_value_).flatten()))
            target_network_sds.append(np.std(np.asarray(target_network_value_).flatten()))
    np.savetxt('mean_cooperation.csv',sample_means,delimiter=',')
    np.savetxt('std_cooperation.csv',sample_sds,delimiter=',')
    np.savetxt('mean_value_action.csv',action_network_means,delimiter=',')
    np.savetxt('mean_value_target.csv',target_network_means,delimiter=',')
    np.savetxt('std_value_action.csv',action_network_sds,delimiter=',')
    np.savetxt('std_value_target.csv',target_network_sds,delimiter=',')

def make_graphs():
    mean_cooperation = genfromtxt('mean_cooperation.csv',delimiter=',')
#    std_cooperation = genfromtxt('std_cooperation.csv',delimiter=',')
    mean_value_action = genfromtxt('mean_value_action.csv',delimiter=',')
#    mean_value_target = genfromtxt('mean_value_target.csv',delimiter=',')
#    std_value_action = genfromtxt('std_value_action.csv',delimiter=',')
#    std_value_target = genfromtxt('std_value_target.csv',delimiter=',')

    x = []
    for i in range(0,80000000,800000):
        x.append(i)

    ne = np.full(100,4)

    plt.plot(x,mean_value_action,x,ne)
    plt.xlabel('samples')
    plt.ylabel('mean action value estimate')
    plt.title('Mean action value estimate vs samples')
    plt.legend(['Agent value estimate','NE value'])
    plt.show()
    plt.close()
    plt.plot(x,mean_cooperation)
    plt.xlabel('samples')
    plt.ylabel('mean cooperation')
    plt.title('Mean cooperation vs number of samples')
    plt.show()


def sem():
    std_cooperation = genfromtxt('std_cooperation.csv',delimiter=',')
    std_value_action = genfromtxt('std_value_action.csv',delimiter=',')
    # standard error of the mean is computed as the standard deviation of the sampling distribution/sqrt(number of samples)
    std_cooperation = std_cooperation/10. #there are 100 samples
    std_value_action = std_value_action/10.

    x = []
    for i in range(0,80000000,800000):
        x.append(i)

    with open('sem_cooperation.csv','ab') as f:
#        np.savetxt(f,np.asarray(x),delimiter=',')
        np.savetxt(f,std_cooperation,delimiter=',')

    with open('sem_value_action.csv','ab') as f:
#        np.savetxt(f,np.asarray(x),delimiter=',')
        np.savetxt(f,std_value_action,delimiter=',')
#    np.savetxt('sem_cooperation.csv',np.column_stack((np.asarray(x),std_cooperation)))
#    np.savetxt('sem_value_action.csv',np.column_stack((np.asarray(x),std_value_action)))

state = tf.placeholder('float32',[None,2*args.r,2*args.r,1],name='state')
y = tf.placeholder('float32',[None],name='y')

qnet_output = qnet(args,state,False)
qnet_value = tf.reduce_max(qnet_output,axis=1)
qnet_best_value = tf.nn.top_k(qnet_output,k=16)[0][:,0] #best value now
qnet_prob = tf.nn.softmax(qnet_output)
qnet_choice = tf.argmax(qnet_output,axis=1)
qnet_loss = tf.reduce_mean(tf.squared_difference(y,qnet_best_value))
qnet_opt = tf.train.AdamOptimizer(learning_rate=args.lr)
qnet_grads = qnet_opt.compute_gradients(qnet_loss)
qnet_train = qnet_opt.apply_gradients(qnet_grads)
qnet_norm = tf.global_norm([i[0] for i in qnet_grads])

target_net_output = target_net(args,state,False)
target_net_value = tf.reduce_max(target_net_output,axis=1)
target_net_loss = tf.reduce_mean(tf.squared_difference(y,target_net_value)) 
target_net_opt = tf.train.AdamOptimizer(learning_rate=args.lr)
target_net_grads = target_net_opt.compute_gradients(target_net_loss)
target_net_train = target_net_opt.apply_gradients(target_net_grads)
target_net_norm = tf.global_norm([i[0] for i in target_net_grads])

init_weights = tf.global_variables_initializer()
saver = tf.train.Saver()

if args.train_model:
    with tf.Session() as sess:
        sess.run(init_weights)
        for i in range(args.epochs): # each batch includes 400 samples, 1 batchs per epoch, 400 samples per epoch, 200,000 epochs, 80M samples
            state_t_ = init(args)#[:,:,0]

            batch_t_ = batch(args.dim,args.r,state_t_)
            r_t_ = payoff(state_t_)
            qnet_output_t_ = (sess.run([qnet_choice],feed_dict={state:batch_t_})[0])
            state_t1_ = next_state(args,qnet_output_t_,True)
            batch_t1_ = batch(args.dim,args.r,state_t1_)
            qnet_output_t1_ = sess.run([qnet_choice],feed_dict={state:batch_t1_})[0]
            qnet_value_t_ = sess.run([qnet_output],feed_dict={state:batch_t_})[0]
            qnet_value_t1_ = sess.run([target_net_output],feed_dict={state:batch_t1_})[0] 
            qnet_prob_t1_ = sess.run([qnet_prob],feed_dict={state:batch_t1_})[0]
            label = r_t_ + args.gamma*np.sum((np.multiply(qnet_value_t1_,qnet_prob_t1_)),axis=1)
            _,qnet_loss_,qnet_norm_,qnet_choice_ = sess.run([qnet_train,qnet_loss,qnet_norm,qnet_choice],feed_dict={state:batch_t_,y:label})
            _,target_net_loss_,target_net_norm_ = sess.run([target_net_train,target_net_loss,target_net_norm],feed_dict={state:batch_t_,y:label})
            q_max_t_ = np.amax(qnet_value_t_,axis=1)
            print('t loss',target_net_loss_,'t gradient',target_net_norm_)
            print('label',label)#,'q worst',q_2max_t_)
            print('epoch',i,'q loss',qnet_loss_,'q gradient',qnet_norm_,'q out',qnet_output_t_,'q value',qnet_value_t_,'action max',q_max_t_)#,'max',q_,'choice',choice_)#'q',q_,'choice',choice_)#,'q',q_,'choice',choice_)#,'\n','label',label_,'\n','qt+1',q_t1_,'\n','r_t',r_t)#,'out',raw,'label',label,'choice',choice_)
            state_t_ = state_t1_
            if i%2000==0:
                saver = tf.train.Saver()
                saver.save(sess,'/save_model/iter'+str(int(i/2))+'.ckpt')

else:
    with tf.Session() as sess:
        saver.restore(sess,'final/save_model/'+args.model)
        print('restored')
        test_ = init(args)
        visualise(test_,'0')
        for i in range(1,args.test_steps):
            print(i)
            batch_test_ = batch(args.dim,args.r,test_)
            out_test_ = sess.run([qnet_choice],feed_dict={state:batch_test_})[0]
            test_t1_ =  next_state(args,out_test_,False)
            visualise(test_t1_,str(i))
            test_ = test_t1_




