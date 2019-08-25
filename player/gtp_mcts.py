import sys
import argparse
import os
import logging
import numpy as np

from gobase import GoModel
from mcts import MCTS

def play(mcts,color,pos):
    player=0 if color[0]=='b' else 1
    if mcts.state.get_total_moves()%2 != player:
        logging.error('wrong player color %s'%color) 
        return

    if pos=='pass':
        action=-1
    else:
        col=(ord(pos[0])-ord('A')) if pos[0] != 'J' else 8;
        row=ord(pos[1])-ord('1')
        action=row*9+col 

    mcts.apply_move(action)

def genmove(mcts,color,model):
    player=0 if color[0]=='b' else 1
    if mcts.state.get_total_moves()%2 != player:
        logging.error('wrong player color %s'%color) 
        exit()

    action,dump=mcts.genmove()
    sys.stderr.write(dump)

    if action==-1:
        pos='pass'
    else:
        col=action%9
        row=action/9
        pos=('J' if col==8 else chr(ord('A')+col))+(chr(ord('1')+row))

    logging.info('action=%s'%pos)
    #logging.info('dump:\n%s'%dump)
    return pos

def gtp_print(ret=''):
    sys.stdout.write('= %s\n\n'%ret)
    logging.info('return %s'%ret) 
    sys.stdout.flush()

def process():
    model = GoModel(args)
    model.build()
    model.load_model()

    mcts=MCTS(model)
    last_color=None

    while True:
        line=sys.stdin.readline().strip()
        if len(line)==0:
            continue
        logging.info('receive command %s'%line)
        splits=line.strip().split()
        cmd=splits[0]
        if cmd=='play':
            color=splits[1]
            pos=splits[2]
            play(mcts,color,pos)

            gtp_print()
            last_color=color
        elif cmd=='genmove':
            color=splits[1]
            if last_color==color:
                #implicit opponent pass        
                play(mcts,'white' if color[0]=='b' else 'black','pass')
            pos=genmove(mcts,color,model)

            gtp_print(pos)
            mcts.print_board()
            last_color=color
        elif cmd=='time_left':
            gtp_print()
        elif cmd=='name':
            gtp_print('weak_bot')
        elif cmd=='quit':
            gtp_print()
            return
        elif cmd=='protocol_version':
            gtp_print('0.1')
        elif cmd=='version':
            gtp_print('1.0')
        elif cmd=='list_commands':
            gtp_print("protocol_version\nname\nversion\nlist_commands\nquit\nboardsize\nclear_board\nkomi\nplay\ngenmove\nfinal_score\ntime_left\ntime_settings\ncgos-opponent_name\ncgos-gameover\n")
        elif cmd=='boardsize':
            gtp_print()
        elif cmd=='clear_board':
            gtp_print()
        elif cmd=='komi':
            gtp_print()
        elif cmd=='final_score':
            s=mcts.state.get_score_black()
            if s>0:
                score='B+%.1f'%s
            else:
                score='W+%.1f'%-s
            gtp_print(score)
        elif cmd=='time_settings':
            gtp_print()
        elif cmd=='cgos-opponent_name':
            gtp_print()
            op_name=splits[1]
            logging.info("--------------------------start of game--------------------------------");
            logging.info("opponent name=%s"%op_name);
        elif cmd=='cgos-gameover':
            gtp_print()
            result=splits[1]
            logging.info("game ended result=%s"%result);
            logging.info("--------------------------end of game  --------------------------------");
        else:
            logging.error('Unknow command %s'%line) 
            gtp_print()

if __name__=='__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--model_path', type=str, default=None)
    argparser.add_argument('--log',type=str,default='mcts.log')

    args = argparser.parse_args()

    logging.basicConfig(filename=args.log, filemode="w", format="%(asctime)s %(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    process()
