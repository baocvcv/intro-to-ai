#include <iostream>
#include <cstdlib>
#include <ctime>

#include "Node.h"

using namespace std;

void Node::init (
        int p, const int s[MAX_BOARD_SIZE][MAX_BOARD_SIZE],
        int M, int N, const MyPoint& cur_move, const int* top, const int color,
        const int noX, const int noY) {
    this->parent = p;
    this->a = cur_move;
    this->color = color;
    this->noX = noX, this->noY = noY;
    nrow = M, ncol = N;
    for (int j = 0; j < ncol; j++) {
        for (int i = 0; i < nrow; i++)
            (this->s)[i][j] = s[i][j]; // copy over the current board
        (this->top)[j] = top[j];
        if (next_top(j) >= 0) {
            children[j] = -1; // set to available
            max_child_num++;
        } else {
            children[j] = -2;
        }
    }
    if (max_child_num == 0) type = -3;
}

void Node::init_with_move (
        int p, const int old_s[MAX_BOARD_SIZE][MAX_BOARD_SIZE],
        int M, int N, const MyPoint& cur_move, const int* old_top, const int color,
        const int noX, const int noY) {
    init(p, old_s, M, N, cur_move, old_top, color, noX, noY);
    (this->s)[a.first][a.second] = color;
    (this->top)[a.second] = a.first; // make the move
    if (next_top(a.second) < 0) {
        max_child_num--;
        children[a.second] = -2;
    }
    if (max_child_num == 0) type = -3;
}

void Node::expand(int this_idx, int child_idx, Node& child) {
    int i;
    if (type >= 0) {
        i = the_move;
        children[i] = child_idx;
    } else
        for (i = 0; i < ncol; i++)
            if (children[i] == -1) { // found a node to expand
                children[i] = child_idx;
                break;
            }
    child.init_with_move(this_idx, s, nrow, ncol, {next_top(i), i},
                         top, flip(color), noX, noY);
    child_num++;
    if (child_num >= max_child_num) type = -3;
}

//TODO: how does nox, noy affect marking and evaluating?
const int Node::dirx[8] = {0, 0,-1, 1, 1,-1, 1, 0};
const int Node::diry[8] = {0,-1,-1,-1, 0, 1, 1, 1};
void Node::mark_location(int s_cur[][12][8], int x, int y) {
    for (int dir = 1; dir < 8; dir++) { // each dir
        int x1 = x + dirx[dir], y1 = y + diry[dir];
        bool in_range = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
        if (!in_range || s_cur[x1][y1][0] == 0) {
            s_cur[x][y][dir] = 0;
            continue;
        } else { s_cur[x][y][dir] = 1; }

        int p_color = s_cur[x1][y1][0];
        for (int i = 2; i < 12; i++) {
            x1 = x + i*dirx[dir], y1 = y + i*diry[dir];
            in_range = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
            if (in_range && s_cur[x1][y1][0] == p_color)
                s_cur[x][y][dir]++;
            else
                break;
        }
    }
}

void Node::evaluate_strength(int s_cur[][12][8], int top_cur[][2], int turn, int x, int y) {
    top_cur[y][1] = 0; // reset
    for (int dir = 1; dir < 8; dir++) { // consider one dir only
        int x1 = x + dirx[dir], y1 = y + diry[dir];
        bool in_range1 = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
        if (in_range1) {
            if (s_cur[x][y][dir] >= 3) {
                if (s_cur[x1][y1][0] == turn) {
                    top_cur[y][1] = 3;
                    return;
                } else { top_cur[y][1] = 2; }
            } else if (s_cur[x][y][dir] > 0) {
                top_cur[y][1] = max(1, top_cur[y][1]);
            }
        }
    }
    for (int dir = 1; dir < 4; dir++) { // for each pair of dirs
        int x1 = x + dirx[dir], y1 = y + diry[dir];
        int x2 = x + dirx[8-dir], y2 = y + diry[8-dir];
        bool in_range1 = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
        bool in_range2 = 0 <= x2 && x2 < nrow && 0 <= y2 && y2 < ncol;
        if (in_range1 && in_range2) { // both dir available
            if (s_cur[x1][y1][0] == s_cur[x2][y2][0]) { // same color
                int strength = s_cur[x][y][dir] + s_cur[x][y][8-dir];
                if (strength >= 3) { // 3 or 2
                    if (s_cur[x1][y1][0] == turn) { // 3
                        top_cur[y][1] = 3;
                        return;
                    } else { top_cur[y][1] = 2; }
                // TODO: is this reasonable?
                } else if (strength > 0) {
                    top_cur[y][1] = max(1, top_cur[y][1]);
                }
            }
        }
    }
}

int Node::defaultPolicy() {
    // copy current board: s_cur[i][j][x]
    // x = 0: the board; 1-7: count the num of consecutive pieces of each direction
    // 1 left, 7 right; 2 up left, 6 down right; 3 down left, 5 up right; 4 down
    int s_cur[12][12][8];
    // top_cur[i][0] is top, top_cur[i][1] is the strength of the move
    // 3 - I win, 2 - opponent wins, 1 - good, 0 - average
    int top_cur[12][2];
    for (int j = 0; j < ncol; j++) {
        top_cur[j][0] = top[j];
        for (int i = 0; i < nrow; i++)
            s_cur[i][j][0] = s[i][j];
    }
    int turn = flip(color); // whose turn it is for the current move

    // mark the map
    for (int y = 0; y < ncol; y++) {
        if (next_top(top_cur, y) < 0) continue;
        int x = next_top(top_cur, y);
        mark_location(s_cur, x, y);
        evaluate_strength(s_cur, top_cur, turn, x, y);
    }
    // check for forcing moves
    for (int i = 0; i < ncol; i++) {
        if (next_top(top_cur, i) < 0) continue;
        if (top_cur[i][1] == 3) { // winnable
            type = -1, the_move = i;
            break;
        } else if (top_cur[i][1] == 2) {
            type = the_move = i;
        }
    }

    auto is_draw = [&] {
        for (int i = 0; i < ncol; i++)
            if (next_top(top_cur, i) >= 0)
                return false;
        return true;
    };
    // simulate until one side wins
    ::srand(time(0));
    bool is_ended = false;
    bool is_drawn = false;
    while (!is_ended) {
        if ((is_drawn = is_draw())) break; // check for draw

        // check if a certain move wins or prevents loss
        int y = -1;
        bool has_forcing_moves = false;
        bool has_good_moves = false;
        for (int i = 0; i < ncol; i++) { // check for forcing moves
            if (next_top(top_cur, i) < 0) continue;
            if (top_cur[i][1] == 3) { // winnable
                is_ended = true;
                has_forcing_moves = true, y = i;
                break;
            } else if (top_cur[i][1] == 2) {
                has_forcing_moves = true, y = i;
            }
        }
        if (!has_forcing_moves) {
            for (int i = 0; i < ncol; i++) { // check for good moves
                if (next_top(top_cur, i) >= 0 && top_cur[i][1] > 0) {
                    has_good_moves = true;
                    break;
                }
            }
            // make a random move among good moves
            int level = has_good_moves ? 1 : 0;
            while (true) { // generate legal move
                y = rand() % ncol;
                if (next_top(top_cur, y) >= 0 && top_cur[y][1] == level)
                    break;
            }
        }
        top_cur[y][0] = next_top(top_cur, y);
        s_cur[top_cur[y][0]][y][0] = turn; // make move
        turn = flip(turn);

        // mark the map again
        if (!is_ended) {
            for (int y_tmp = 0; y_tmp < ncol; y_tmp++) {
                if (top_cur[y_tmp][0] == 0) continue; // column full
                int x = next_top(top_cur, y_tmp);
                // only recalculate the candidate moves that are on the same line
                // as the new move and has a distance of less than 4
                // int distX = abs(top_cur[y][0] - x), distY = abs(y - y_tmp);
                // if ((distX==distY || distX==0 || distY==0) && max(distX,distY)<4) {
                    mark_location(s_cur, x, y_tmp);
                    evaluate_strength(s_cur, top_cur, turn, x, y_tmp);
                // }
            }
        }
    }

    #ifdef DEBUG
    // cerr << "Simulation ---" << endl;
    // for (int i = 0; i < nrow; i++) {
    //     for (int j = 0; j < ncol; j++)
    //         cerr << s_cur[i][j][0] << ' ';
    //     cerr << endl;
    // }
    // cerr << "draw: " << is_drawn << " turn: " << turn << " color: " << color << endl;
    // cerr << endl;
    #endif
    
    if (is_drawn) return 1;
    if (turn == color) // opponent's turn to move when game ended
        return 2;
    else
        return 0;
}

inline int Node::next_top(int y) {
    if (y == noY && top[y]-1 == noX) return noX - 1;
    else return top[y] - 1;
}

inline int Node::next_top(int top_cur[][2], int y) {
    if (y == noY && top_cur[y][0]-1 == noX) return noX - 1;
    else return top_cur[y][0] - 1;
}