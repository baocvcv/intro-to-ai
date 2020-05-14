#ifndef UCT_HPP_
#define UCT_HPP_

#include <utility>
#include <memory>

#include "Config.hpp"
#include "Point.h"
#include "Node.hpp"

class Board;

class UCT {
public:

    static Point* UCTSearch(int _M, int _N, int _noX, int _noY, int lastX, int lastY,
                           int const* _board, const int *_top);

private:
    UCT() {} // disable constructor

    // static int nrow, ncol, noX, noY;

    static Node& treePolicy(Node& root);
    static void backPropagate(Node& node, double gain);
};

#endif