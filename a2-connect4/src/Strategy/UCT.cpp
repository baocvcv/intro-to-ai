#include <ctime>
#include <memory>
#include <cstdio>

#include "UCT.hpp"
#include "Node.hpp"
#include "Board.hpp"
#include "Config.hpp"

// int UCT::nrow = 0;
// int UCT::ncol = 0;
// int UCT::noX = -1;
// int UCT::noY = -1;
// int UCT::top[MAX_BOARD_SIZE] = { 0 };

Point* UCT::UCTSearch(int M, int N, int noX, int noY, int lastX, int lastY,
                     const int* board, const int* top)
{
    auto time0 = clock(); // start timer

    Node& root = Node::init_pool();
    root.init(0, -1, M, N, board, top, 1, lastX, lastY, noX, noY);
    #if defined(DEBUG) || defined(DEBUG2)
        root.print();
    #endif
    if (lastX != -1) // do not emulate when starting a new game
        backPropagate(root, root.defaultPolicy());

    int times = 0;
    #ifdef DEBUG2
    float time_d = 0;
    #else
	float time_d = 1e6 * (clock() - time0) / CLOCKS_PER_SEC;
    #endif
    while (time_d < TIME_LIMIT) {
        times++;
        auto& node = treePolicy(root);
        auto gain = node.defaultPolicy();
        backPropagate(node, gain);
        #ifdef DEBUG2
        time_d += 200;
        #else
	    time_d = 1e6 * (clock() - time0) / CLOCKS_PER_SEC;
        #endif
    }
    auto move = root.bestMove();

    #if defined(DEBUG) || defined(DEBUG2)
        for (int i = 0; i < root.ncol; i++) {
            auto& n = Node::pool[root.children[i]];
            fprintf(stderr, "%d-(%d, %d) ", i, n.N, n.Q);
            if (i == root.ncol / 2 - 1) fprintf(stderr, "\n");
        }
        fprintf(stderr, "\nMove: (%d, %d) @ %d\n\n", move.x, move.y, times);
    #endif
    fprintf(stderr, "Sim %d times\n", times);
    return new Point { move.x, move.y };
}

Node& UCT::treePolicy(Node& root) {
    int idx = root.self;
    while (!Node::pool[idx].is_leaf()) {
        if  (Node::pool[idx].expandable())
            return Node::pool[idx].expand();
        else
            idx = Node::pool[idx].bestChild();
    }
    return Node::pool[idx];
}

void UCT::backPropagate(Node& node, double gain) {
    node.back_propagate(gain);
}