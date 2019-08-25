#ifndef GOBOARD2_H
#define GOBOARD2_H

#include <cassert>
#include <string>
#include <map>
#include <vector>

extern "C" {
#include "board.h"
#include "mq.h"
}

/**
 * The class representing a GO board
 * NOTE: This is a C++ Wrapper class of pachi's board representation in C.
 */
class GoBoard2
{
private:
	struct board *b; /**< A pointer to pachi's GO board representation */
	std::string history;

	// each move 2 chars
	// color is implied in index. the first move is black.
	// normal move: col in A..S, row in a..s
	//     e.g. C4 => Cd
	// pass       : 'ZP'
	// resign     : 'ZR'
	void add_move_to_history(const int pos) {
		char hi, lo;
		if (POS_PASS==pos) {
			hi = 'Z', lo = 'P';
		} else if (POS_RESIGN==pos) {
			hi = 'Z', lo = 'R';
		} else {
            hi = (pos%9)+'A';
            lo = (pos/9)+'a';
		}
		history.append(1,hi);
		history.append(1,lo);
	}

    int _apply_move(int pos,int color);

public:
    static const int COLOR_BLACK=0;
    static const int COLOR_WHITE=1;
    static const int COLOR_EMPTY=-1;

    static const int POS_PASS=-1;
    static const int POS_RESIGN=-2;

    static const int MOVE_SUCC=0;
    static const int MOVE_INVALID=1;
    static const int MOVE_SUPERKO=2;

	GoBoard2(){
        b=board_init();
        char rule[100]="chinese";
        board_set_rules(b, rule);
        board_resize(b,9);
        board_clear(b);
        b->komi=7.5;
        history.clear();
    }

	GoBoard2(const GoBoard2 &other){
        b = (board *) malloc(sizeof(*b));
        board_copy(b, other.b);
        history = other.history;
    }

	~GoBoard2() {
		board_done(b);
	}

    void reset(){
        board_clear(b);
        history.clear();
    }

	void print_board() {
        print_board(stderr);
        fprintf(stderr, "history:%s\n",get_history().c_str());
	}

    void print_board(FILE * f){
        board_print(b,f);
    }

	/**
	 * Returns the color of next player.
	 */
	int get_color() {
        return (history.length()%4)>>1;
	}

	float get_score_black() {
        return -board_official_score(b, NULL);
	}
    
    bool is_pass_safe(){
        float score = board_official_score(b, NULL);
        if (get_color() == COLOR_BLACK)
            score = -score;
        return (score >= 0);
    }

	std::string get_history() const {
		return history;
	}
	
	int get_total_moves() {
		return history.length()/2;
	}

	int apply_move(int action); 
	bool apply_history(const std::string& history);
    void sensible_moves(int* mask, int my_color_int);
    void valid_moves(int* mask, int my_color_int);

    void extract_feature(float* feature);
};

#endif
