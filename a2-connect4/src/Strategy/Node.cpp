#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <limits>

#include "Node.hpp"

using namespace std;

const Point NO_MOVE({ -1, -1 });
const int Node::dirx[8] = {0, 0,-1, 1, 1,-1, 1, 0};
const int Node::diry[8] = {0,-1,-1,-1, 0, 1, 1, 1};
std::unique_ptr<Node[]> Node::pool(new Node[MAX_POOL_SIZE]);
int Node::node_cnt = 0;

Node& Node::init_pool() {
    // pool.reset(new Node[MAX_POOL_SIZE]);
    node_cnt = 1;
    return pool[0];
}

void Node::init(int self, int parent, int M, int N, const int* _board, const int* _top,
                int lastColor, int lastX, int lastY, int noX, int noY)
{
    this->self = self;
    this->parent = parent;
    this->last_move = Point(lastX, lastY);
    this->last_color = lastColor;
    this->noX = noX, this->noY = noY;
    Q = 0; this->N = 0;
    max_child_num = child_num = 0;
    nrow = M; ncol = N;
    type = NodeType::FREELY_EXPANDABLE;

    board = Board(M, N, _board);
    for (int j = 0; j < ncol; j++) {
        (this->top)[j] = _top[j];
        if (next_top(j) >= 0) {
            children[j] = -1; // set to available
            max_child_num++;
        } else {
            children[j] = -2;
        }
    }
    if (max_child_num == 0) type = NodeType::FULL;
    if (lastX != -1)
        set_node_type();
}

void Node::init_with_move (int self, int parent, int M, int N, const Board& _board, const short* _top,
                           int lastColor, Point last_move, int noX, int noY)
{
    this->self = self;
    this->parent = parent;
    this->last_move = last_move;
    this->last_color = lastColor;
    this->noX = noX, this->noY = noY;
    Q = 0; this->N = 0;
    max_child_num = child_num = 0;
    nrow = M; ncol = N;
    type = NodeType::FREELY_EXPANDABLE;

    board = _board;
    board.piece_at(last_move.x, last_move.y, lastColor);
    for (int j = 0; j < ncol; j++) {
        (this->top)[j] = (j == last_move.y) ? last_move.x : _top[j];
        if (next_top(j) >= 0) {
            children[j] = -1; // set to available
            max_child_num++;
        } else {
            children[j] = -2;
        }
    }
    if (max_child_num == 0) type = NodeType::FULL;
    set_node_type();
}

void Node::set_node_type() {
    auto move = get_forcing_move(board, top, flip(last_color), last_move);
    if (move.first == MoveType::FORCING)
        type = NodeType::FORCING, the_move = move.second;
    else if (move.first == MoveType::LOSING)
        type = NodeType::LOSING, the_move = NO_MOVE;
}

Node& Node::expand() {
    int y;
    if (type == NodeType::FORCING) {
        y = the_move.y;
    } else { // NodeType::FREELY_EXPANDABLE
        for (y = 0; y < ncol; y++) if (children[y] == -1) break;
    }
    pool[node_cnt].init_with_move(node_cnt, self, nrow, ncol, board, top,
                                  flip(last_color), {next_top(y), y}, noX, noY);
    children[y] = node_cnt;
    child_num++;
    if (child_num >= max_child_num) type = NodeType::FULL;
    return pool[node_cnt++];
}

int Node::bestChild() {
    assert(type == NodeType::FULL);
    int best_child = -1;
    double max_score = -std::numeric_limits<double>::max();
    for (int i = 0; i < ncol; i++) {
        if (children[i] < 0) continue;
        double score = pool[children[i]].calc_score(N);
        if (score > max_score) {
            max_score = score;
            best_child = children[i];
        }
    }
    if (best_child < 0) {
        fprintf(stderr, "[Error] best_child returning -1. Aborting...\n");
        fprintf(stderr, "self-%d p-%d max_c-%d c_no-%d\n", self, parent, max_child_num, child_num);
    }
    return best_child;
}

Point Node::bestMove() {
    if (is_leaf()) {
        for (int i = 0; i < ncol; i++) {
            if (children[i] >= -1) {
                return { next_top(i), i };
            }
        }
    }
    return pool[bestChild()].last_move;
}

int Node::defaultPolicy() {
    if (type == NodeType::LOSING) return PROFIT_LOSS;

    // calc remaining moves
    int remaining_moves = 0;
    for (int y = 0; y < ncol; y++) {
        if (y != noY) remaining_moves += top[y];
        else if (top[y] > noX) remaining_moves += top[y] - 1;
    }

    short top_cur[12];
    for (int j = 0; j < ncol; j++) {
        top_cur[j] = top[j];
    }
    Board board_cur = board;
    int turn = last_color; // last move done by last_color
    auto last_move_cur = last_move;

    // simulate until one side wins
    ::srand(time(0));
    bool is_ended = board_cur.is_won(last_color) || remaining_moves <= 0;
    while (!is_ended) {
        turn = flip(turn); // switch turn
        auto move = get_forcing_move(board_cur, top_cur, turn, last_move_cur);

        if (move.first == MoveType::AVERAGE) {
            int y = rand() % ncol;
            while (top_cur[y] == 0) y = rand() % ncol;
            move.second = Point({ top_cur[y]-1, y });
        } else if (move.first == MoveType::LOSING) {
            return (turn == last_color) ? PROFIT_WIN : PROFIT_LOSS;
        }

        board_cur.place(move.second.x, move.second.y, turn);
        top_cur[move.second.y]--, remaining_moves--;
        if (move.second.y == noY && move.second.x-1 == noX)
            top_cur[move.second.y]--, remaining_moves--;

        is_ended = board_cur.is_won(turn) || remaining_moves <= 0;
    }

    if (board_cur.is_won(turn)) {
        if (turn == last_color) // opponent just moved and game ended
            return PROFIT_LOSS;
        else if (turn == flip(last_color))
            return PROFIT_WIN;
    }
    return PROFIT_DRAW;
}

Node::Move Node::get_forcing_move(Board& board_cur, short* top_cur, int turn_cur, Point last_move) {
    int color_to_check = flip(turn_cur);
    // check downwards
    int down;
    for (down = last_move.x + 1; down < nrow; down++) {
        if (!board_cur.piece_at(down, last_move.y, color_to_check)) break;
    }
    if (last_move.x - down == 3 && (last_move.y != noY || last_move.x-1 != noX))
        return { MoveType::FORCING, {last_move.x-1, last_move.y}};
    
    // check other dirs
    for (int dir = 1; dir <= 3; dir++) {
        int x1, y1;
        for (x1 = last_move.x+dirx[dir], y1 = last_move.y+diry[dir];
             0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
             x1 += dirx[dir], y1 += diry[dir])
        {
            if (!board_cur.piece_at(x1, y1, color_to_check)) break;
        }
        int x2, y2;
        for (x2 = last_move.x+dirx[8-dir], y2 = last_move.y+diry[8-dir];
             0 <= x2 && x2 < nrow && 0 <= y2 && y2 < ncol;
             x2 += dirx[8-dir], y2 += diry[8-dir])
        {
            if (!board_cur.piece_at(x2, y2, color_to_check)) break;
        }

        if (y2 - y1 >= 4) { // 3 pieces in between
            bool empty1 = (0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol) && top_cur[y1] == x1 + 1;
            bool empty2 = (0 <= x2 && x2 < nrow && 0 <= y2 && y2 < ncol) && top_cur[y2] == x2 + 1;
            if (empty1 && empty2)
                return { MoveType::LOSING, NO_MOVE };
            else if (empty1)
                return { MoveType::FORCING, { x1, y1 } };
            else if (empty2)
                return { MoveType::FORCING, { x2, y2 } };
        }
    }
    return Node::Move({ Node::MoveType::AVERAGE, NO_MOVE });
}

inline int Node::next_top(int y) {
    if (y == noY && top[y]-1 == noX) return noX - 1;
    else return top[y] - 1;
}

// inline int Node::next_top(int top_cur[][2], int y) {
//     if (y == noY && top_cur[y][0]-1 == noX) return noX - 1;
//     else return top_cur[y][0] - 1;
// }