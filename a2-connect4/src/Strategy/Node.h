#ifndef NODE_H_
#define NODE_H_

#include <cmath>

typedef std::pair<int, int> MyPoint; 

const double c = 0.5;
const int MAX_BOARD_SIZE = 12;

class Node {
public:
	Node (): parent(-1), child_num(0), max_child_num(0), Q(0), N(0) {}

    // init
	void init (int p, const int s[MAX_BOARD_SIZE][MAX_BOARD_SIZE], int M, int N,
               const MyPoint& cur_move, const int* top, const int color);
    // init and make the move denoted by a
	void init_with_move (int p, const int old_s[MAX_BOARD_SIZE][MAX_BOARD_SIZE],
                         int M, int N, const MyPoint& cur_move, const int* old_top,
                         const int color);

    // add node child to the list of children
    void expand(int this_idx, int child_idx, Node& child);
    // return global index of the best child
    // return simulation result
    int defaultPolicy();
    // update current node using gain, update bestChild too
    // child_idx is the index of the child from which the update comes
    void update(int gain, int child_idx, const Node& child);

    inline bool is_leaf() { return max_child_num==0; }
    inline bool expandable() { return child_num < max_child_num; }
    inline void update(int gain) { N++, Q += gain; }
    inline double calc_value(int N_p) const {
        return (2*N-Q) * 1.0 / (2*N) + c * sqrt(2 * log(N_p) / N);
    }
    inline int flip(int t) { return 3 - t; }

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
	MyPoint a; // last move
	int color; // color of the last move: 1-opponent's move, 2-my move
	int Q; // Q: simulation wins of side "color"
    int N; // N: total num of simulations
};

#endif