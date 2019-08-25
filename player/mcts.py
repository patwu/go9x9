import numpy as np
import sys
import argparse
import os
from gobase import GoModel

from pyboard import PyBoard


class Node(object):
    def __init__(self, fa, prior, value, sensible):
        self.name=-999
        self.fa = fa
        self.child = {}  
        self.prior = []
        self.vl = {}
        self.n = 0
        self.w = 0
        self.value=value
        self.total_visit=0
        self.saction=[]
        self.act_ind_map={}

        for i,s in enumerate(sensible):
            if prior[s]>0.01:
                self.act_ind_map[s]=len(self.saction)
                self.prior.append(prior[s])
                self.saction.append(s)

        #pass
        self.act_ind_map[-1]=len(self.saction)
        self.saction.append(-1)
        self.prior.append(prior[-1])

        self.saction_size=len(self.saction)
        self.prior=np.asarray(self.prior,dtype=float)

    def select(self, minmax,ignore_pass):
        self.total_visit+=1
        
        u = self.prior * np.sqrt(self.total_visit)
        q = np.zeros(len(self.saction))
        for ind, child in self.child.iteritems():
            u[ind] /= (1+child.n) 
            if ind in self.vl:
                q[ind] = (child.w-self.vl[ind]*minmax) / (child.n+self.vl[ind])
            else:
                q[ind] = child.w / child.n
        #penalty for early pass
        d=u+q*minmax
        if ignore_pass:
            d[-1]=-1
        ind = np.argmax(d)

        if ind in self.vl:
            self.vl[ind]+=1
        else:
            self.vl[ind]=1
        if ind in self.child:
            return self.child[ind],self.saction[ind]
        else:
            return None,self.saction[ind]

    def expand(self, action, value, child_prior, sensible_mask):
        new_child=Node(self,child_prior,value,sensible_mask)
        new_child.name=action
        self.child[self.act_ind_map[action]]=new_child
        return new_child

    def clear_vl(self):
        if self.vl=={}:
            return
        
        for i,_ in self.vl.iteritems():
            self.child[i].clear_vl()
        self.vl={}

    def select_action(self):
        argmax_n=-1
        max_n=-9999
        for ind,child in self.child.iteritems(): 
            if child.n>max_n:
                max_n=child.n
                argmax_n=ind
        return self.saction[argmax_n]
    
    def dump_tree(self,depth=0):
        str_=''
        for ind, child in self.child.iteritems():
            if child.n>20:
                if self.saction[ind]==-1:
                    pos='ZP'
                else:
                    col=self.saction[ind]%9
                    row=self.saction[ind]/9
                    pos=('J' if col==8 else chr(ord('A')+col))+(chr(ord('1')+row))
                str_+=' '*depth+'[act=%s,n=%d,w/n=%.3f,value=%.3f]\n'%(pos,child.n,child.w*1./child.n,child.value)
                str_+=child.dump_tree(depth+1)
        return str_

class MCTS(object):
    def __init__(self,model,history='',eval_batchsize=4,n_rollout=800):
        self.state=PyBoard()
        self.state.apply_history(history)
        self.root=None
        self.pending=[]
        self.terminate=[]
        self.model=model
        self.n_rollout=n_rollout
        self.eval_batchsize=eval_batchsize
        self.board_size=9
        self.pass_thres=50

    def init_root(self):
        features=[self.state.extract_feature()] 
        child_prior,value=self.model.predict(features)
        minmax=1 if self.state.get_total_moves()%2==0 else -1
        sensible=self.state.get_valid_move()
        self.root=Node(None,child_prior[0],value[0]*minmax,sensible)
        self.root.color=minmax
        self.root.n=1
        self.root.w=value[0]*minmax

    def select_path(self,board):
        cur=self.root
        total_moves=board.get_total_moves()
        minmax=1

        while True:
            nxt,action = cur.select(minmax,total_moves<self.pass_thres)
            stat = board.apply_move(action)
            if nxt is None or board.is_terminate():
                break
            cur=nxt
            total_moves+=1
            minmax*=-1
            
        if board.is_terminate():
            self.terminate.append([cur,action,board])
        else:
            for fa,a,_ in self.pending:
                if fa==cur and a==action:
                    return
            self.pending.append([cur,action,board])
    
    def backup_terminate(self):
        for i in range(len(self.terminate)):
            cur,action,board=self.terminate[i]
            result=1 if board.get_score_black() > 0 else -1
            if self.state.get_total_moves()%2==1:
                result*=-1
            cur=cur.expand(action,result,np.zeros(self.board_size**2+1),[])
            while not cur is None:
                cur.w+=result
                cur.n+=1
                cur=cur.fa

    def backup_pending(self,nn_result):
        priors,values=nn_result
        for i in range(len(self.pending)):
            cur,action,board = self.pending[i]
            child_prior,result = priors[i],values[i]
            if board.get_total_moves()%2 != self.state.get_total_moves()%2:
                result*=-1
            
            sensible=board.get_valid_move()
            cur=cur.expand(action,result,child_prior,sensible)
            while not cur is None:
                cur.w+=result
                cur.n+=1
                cur=cur.fa 

    def clear(self):
        self.root.clear_vl()
        self.pending=[]
        self.terminate=[]
        self.idle_pending=0
        self.idle_terminal=0

    def genmove(self):
        rollout=0
        rollout_limit=self.n_rollout

        self.init_root()
        cnt=0
        while rollout<rollout_limit or not len(self.pending)+len(self.terminate)==0:
            copy = PyBoard(self.state)
            self.select_path(copy)
            cnt+=1

            if cnt==4:
                cnt=0
                rollout+=len(self.pending)+len(self.terminate)
                features=[board.extract_feature() for _,_,board in self.pending]
                if len(features)!=0:
                    nn_result=self.model.predict(features)
                    self.backup_pending(nn_result)
                
                self.backup_terminate()
                self.clear()
        
        action=self.root.select_action()
        dump=self.root.dump_tree(0)
        self.state.apply_move(action)
        self.root=None
        return action,dump

    def print_board(self):
        self.state.print_board()

    def apply_move(self,move):
        self.state.apply_move(move)

    def is_terminate(self):
        if self.state.is_terminate():
            return True
        else:
            return False

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--model_path', type=str, default='go-models')
    argparser.add_argument('--log',type=str,default='test.log')
    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    model = GoModel(args)
    model.build()
    model.load_model()

    mcts=MCTS(model)#,history='FdDfEfEgFfFgGgDgDdCcCdBdBeBcCeEcDcDbFbEbEdFcBgGhHhGfHgChBhEhHfDiFiGiBiCgHiFhFeGdHdHcGeCiAfGbGcCfBfGdAhCbIdAdGcHbAeAbIhEi')
    
    
    while True:
        print '*'*40
        if mcts.is_terminate():
            break
        a,log=mcts.genmove()
        print log
        mcts.print_board()




