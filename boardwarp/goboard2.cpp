#include <algorithm>
#include <cmath>
#include "goboard2.h"

extern "C" {
#include "board.h"
}

int GoBoard2::apply_move(int action) {
    int succ=GoBoard2::MOVE_SUCC;
    if(action>=0){
        succ=_apply_move(action,get_color());
    }
    add_move_to_history(action);
    return succ;
}

int GoBoard2::_apply_move(int action, int color) {
    struct move m;
    m.color = color==COLOR_BLACK ? S_BLACK : S_WHITE;
    m.coord = coord_xy(b,action%9+1,action/9+1);
    int succ=GoBoard2::MOVE_SUCC;
    if (board_is_valid_play_no_suicide(b, m.color, m.coord)){
        board_play(b, &m);
        if(b->superko_violation){
            succ=GoBoard2::MOVE_SUPERKO;
        }
    }else{
        succ=GoBoard2::MOVE_INVALID;
    }
    return succ;
}


bool GoBoard2::apply_history(const std::string& history) {
    assert(history.length() % 2 == 0); 
    this->history=history;
    for (std::string::size_type step = 0; step < history.length() / 2; ++step){
        char hi=history[2*step];
        char lo=history[2*step+1];

        if (hi!='Z') {
            int pos=(hi-'A')+(lo-'a')*9;
            int succ=_apply_move(pos,step%2);
            
            if(succ!=GoBoard2::MOVE_SUCC){
                print_board();
                fprintf(stderr, "%d %s %lu\n", succ, history.c_str(), step);
                return false;
            }
        }
    }
    return true;
}

void GoBoard2::sensible_moves(int *mask,int my_color_int){
    enum stone my_color=(my_color_int==COLOR_BLACK?S_BLACK:S_WHITE);
    memset(mask,0,sizeof(int)*81);
    foreach_free_point(b){
        int x=coord_x(c,b);
        int y=coord_y(c,b);
        if(board_is_valid_play_no_suicide(b, my_color, c)){
            if (!board_is_eyelike(b, c, my_color) || board_is_false_eyelike(b, c, my_color)){
                mask[(x+y*9-10)]=1;
            }
        }
    }foreach_free_point_end;
}

void GoBoard2::valid_moves(int *mask,int my_color_int){
    enum stone my_color=(my_color_int==COLOR_BLACK?S_BLACK:S_WHITE);
    memset(mask,0,sizeof(int)*81);
    foreach_free_point(b){
        int x=coord_x(c,b);
        int y=coord_y(c,b);
        if(board_is_valid_play_no_suicide(b, my_color, c)){
            mask[(x+y*9-10)]=1;
        }
    }foreach_free_point_end;
}

void GoBoard2::extract_feature(float* feature){
    int stride=10;
    memset(feature,0,sizeof(float)*9*9*stride);

    //last move
    int history_len=history.length();
    for(int turn_since=0; turn_since<std::min(4,int(history_len/2)); turn_since++){
        int col=history[history_len-turn_since*2-2]-65;
        int row=history[history_len-turn_since*2-1]-97;
        if(col!=25){
            //not pass
            feature[(row*9+col)*stride+turn_since]=1.;
        }
    }
    
    //stone
    enum stone my_color=(history_len%4==0?S_BLACK:S_WHITE);
    int offset=4;
    foreach_point(b){
        enum stone color = board_at(b, c);
        if (color!=S_OFFBOARD){
            int x=coord_x(c,b);
            int y=coord_y(c,b);
            if(color==S_NONE){
                feature[(x+y*9-10)*stride+offset+2]=1.;
            }else if (color==my_color){
                feature[(x+y*9-10)*stride+offset]=1.;
            }else{
                feature[(x+y*9-10)*stride+offset+1]=1.;
            }
            feature[(x+y*9-10)*stride+offset+3]=1.;
        }
    }foreach_point_end;


    //color
    offset=8;
    if(my_color==S_BLACK ){
        for(int i=0;i<81;i++){
            feature[i*stride+offset]=1.;
        }
    }else{
        for(int i=0;i<81;i++){
            feature[i*stride+offset]=-1.;
        }
    }

    offset=9;
    //sensible
    foreach_free_point(b){
        int x=coord_x(c,b);
        int y=coord_y(c,b);
        if(board_is_valid_play_no_suicide(b, my_color, c)){
            if (!board_is_eyelike(b, c, my_color) || board_is_false_eyelike(b, c, my_color)){
                feature[(x+y*9-10)*stride+offset]=1;
            }
        }
    }foreach_free_point_end;
}


