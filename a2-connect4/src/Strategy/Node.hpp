#ifndef NODE_HPP_
#define NODE_HPP_

#include <memory>

#include "FastMath.hpp"
#include "Config.hpp"
#include "Point.h"
#include "Board.hpp"

class Node {
public:
    static const int dirx[8], diry[8];
    static std::unique_ptr<Node[]> pool;
    static int node_cnt;

    enum class MoveType {
        AVERAGE,
        FORCING,
        LOSING
    };
    enum class NodeType {
        FULL,
        FREELY_EXPANDABLE,
        FORCING,
        LOSING
    };
    Point the_move; // if is_forcing, the move to make (stores the column)

    using Move = std::pair<MoveType, Point>;
    const static Move AVERAGE;

	Node () {}

    // init pool
    static Node& init_pool();

    // init
	void init (int self, int parent, int M, int N, const int* _board, const int* _top,
               int lastColor, int lastX, int lastY, int noX, int noY);
    // init and make the move denoted by newX, newY
	void init_with_move (int self, int parent, int M, int N, const Board& _board, const short* _top,
               int lastColor, Point last_move, int noX, int noY);

    int bestChild();
    Point bestMove();
    // add node child to the list of children
    Node& expand();
    // true if no more legal moves can be made
    // or game result is certain
    bool is_leaf() {
        return max_child_num==0 ||
               type == NodeType::LOSING;
    }

    void set_node_type();

    void back_propagate(double gain) {
        update(gain);
        int idx = parent;
        while (idx != -1) {
            gain = -gain;
            pool[idx].update(gain);
            idx = pool[idx].parent;
        }
    }
    
    Move get_forcing_move(Board& board_cur, short* top_cur, int turn_cur, Point last_move);

    int expandable() {
        return type == NodeType::FREELY_EXPANDABLE || type == NodeType::FORCING;
    }

    void print() { board.print(); }

    // TODO: fine tune c
    // TODO: better strategy
    // TODO: identify more forcing moves
    //TODO: prevent free 3-in-a-row
    // return global index of the best child
    // return simulation result
    int defaultPolicy();
    // update current node using gain, update bestChild too
    // child_idx is the index of the child from which the update comes
    // void update(int gain, int child_idx, const Node& child);

    void update(int gain) { N++, Q += gain; }
    double calc_score(int N_p) const {
        return -Q*1.0/N + C * fastSqrt(2 * fastLog(N_p) / N);
    }
    int abs(int x) { return (x < 0 ? -x : x); }

    /*
    helper functions
    */
    int flip(int t) { return 3 - t; }

    // void mark_location(int s_cur[][12][8], int x, int y);
    // void evaluate_strength(int s_cur[][12][8], int top_cur[][2], int turn,int x, int y);
    // returns the next available row number that is available in column y
    // if -1, then no more available
    int next_top(int y);
    // int next_top(int top_cur[][2], int y);
    

    int self;
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
    short nrow, ncol;
	short top[12]; // available moves after the last move
    Board board; // board after last_move
    short noX, noY;
	Point last_move; // last move
	short last_color; // color of the last move: 1-opponent's move, 2-my move
	int Q; // Q: simulation wins of side "color"
    int N; // N: total num of simulations
    NodeType type;
};

#endif