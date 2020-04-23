#ifndef NODE_H_
#define NODE_H_

#include <cmath>
#include "config.h"

typedef std::pair<int, int> MyPoint; 

extern const double c;
const int MAX_BOARD_SIZE = 12;

class Node {
public:
	Node (): parent(-1), child_num(0), max_child_num(0),
             Q(0), N(0), type(-2), the_move(-1) {}

    // init
	void init (int p, const int s[MAX_BOARD_SIZE][MAX_BOARD_SIZE], int M, int N,
               const MyPoint& cur_move, const int* top, const int color, const int noX, const int noY);
    // init and make the move denoted by a
	void init_with_move (int p, const int old_s[MAX_BOARD_SIZE][MAX_BOARD_SIZE],
                         int M, int N, const MyPoint& cur_move, const int* old_top,
                         const int color, const int noX, const int noY);

    // TODO: fine tune c
    // TODO: better strategy
    // TODO: identify more forcing moves
    //TODO: prevent free 3-in-a-row
    // add node child to the list of children
    void expand(int this_idx, int child_idx, Node& child);
    // return global index of the best child
    // return simulation result
    int defaultPolicy();
    // update current node using gain, update bestChild too
    // child_idx is the index of the child from which the update comes
    void update(int gain, int child_idx, const Node& child);

    // true if no more legal moves can be made
    // or game result is certain
    bool is_leaf() { return max_child_num==0 || type == -1; }
    void update(int gain) { N++, Q += gain; }
    double calc_win_rate() const { return (N - Q/2.0) / N; }
    double calc_value(int N_p) const {
        return calc_win_rate() + c * sqrt(2 * log(N_p) / N);
    }
    int abs(int x) { return (x < 0 ? -x : x); }
    /*
    -3: all children expanded;
    -2: have children to expand
    -1: have winning move, not expandable
    >=0: only child allowed
    */
    int expandable() { return type; }

    /*
    helper functions
    */
    void mark_location(int s_cur[][12][8], int x, int y);
    void evaluate_strength(int s_cur[][12][8], int top_cur[][2], int turn,int x, int y);
    int flip(int t) { return 3 - t; }
    // returns the next available row number that is available in column y
    // if -1, then no more available
    int next_top(int y);
    int next_top(int top_cur[][2], int y);
    
    static const int dirx[8], diry[8];
	int parent;
	/*
	A parent can have at most 12 children, each corresponding
	to the move made to the column of the same index
	-2: move not available
	-1: move available but not simulated
	>= 0: index of child
	*/
	int children[12];
	int child_num, max_child_num;
    int nrow, ncol;
	int s[12][12]; // state of the board after the last move
	int top[12]; // available moves after the last move
    int noX, noY;
	MyPoint a; // last move
	int color; // color of the last move: 1-opponent's move, 2-my move
	int Q; // Q: simulation wins of side "color"
    int N; // N: total num of simulations
    /*
    -3: not expandable
    -2: expandable
    -1: forcing win, move stored in the_move
    >=0: expandable to only child denoted by type
    */
    int type;
    int the_move; // if is_forcing, the move to make (stores the column)
};

#endif