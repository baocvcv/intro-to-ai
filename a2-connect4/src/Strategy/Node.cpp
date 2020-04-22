#include <iostream>
#include <cstdlib>
#include <ctime>

#include "Node.h"

using namespace std;

void Node::init (
        int p, const int s[MAX_BOARD_SIZE][MAX_BOARD_SIZE],
        int M, int N, const MyPoint& cur_move, const int* top, const int color) {
    this->parent = p;
    this->a = cur_move;
    this->color = color;
    nrow = M, ncol = N;
    for (int j = 0; j < ncol; j++) {
        for (int i = 0; i < nrow; i++)
            (this->s)[i][j] = s[i][j]; // copy over the current board
        (this->top)[j] = top[j];
        if (top[j] > 0) {
            children[j] = -1; // set to available
            max_child_num++;
        } else {
            children[j] = -2;
        }
    }
}

void Node::init_with_move (
        int p, const int old_s[MAX_BOARD_SIZE][MAX_BOARD_SIZE],
        int M, int N, const MyPoint& cur_move, const int* old_top, const int color) {
    init(p, old_s, M, N, cur_move, old_top, color);
    (this->s)[a.first][a.second] = color;
    (this->top)[a.second]--;
    if ((this->top)[a.second] == 0) {
        max_child_num--;
        children[a.second] = -2;
    }
}

void Node::expand(int this_idx, int child_idx, Node& child) {
    int i;
    for (i = 0; i < ncol; i++)
        if (children[i] == -1) { // found a node to expand
            children[i] = child_idx;
            break;
        }
    int newType = flip(color);
    child.init_with_move(this_idx, s, nrow, ncol, {top[i]-1, i}, top, newType);
    child_num++;
}

int dirx[8] = {0, 0,-1, 1, 1,-1, 1, 0};
int diry[8] = {0,-1,-1,-1, 0, 1, 1, 1};
void mark_location(int s_cur[][12][8], int nrow, int ncol, int x, int y) {
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

void evaluate_strength(int s_cur[][12][8], int top_cur[][2], int turn, int nrow, int ncol, int x, int y) {
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
    for (int dir = 1; dir < 3; dir++) { // for each pair of dirs
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

    // helper function
    // dirs: null, left, up-left, down-left, down, up-right, down-right, right
    // auto mark_location = [&](int x, int y) {
    //     for (int dir = 1; dir < 8; dir++) { // each dir
    //         int x1 = x + dirx[dir], y1 = y + diry[dir];
    //         bool in_range = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
    //         if (!in_range || s_cur[x1][y1][0] == 0) {
    //             s_cur[x][y][dir] = 0;
    //             continue;
    //         } else { s_cur[x][y][dir] = 1; }

    //         int p_color = s_cur[x1][y1][0];
    //         for (int i = 2; i < 12; i++) {
    //             x1 = x + i*dirx[dir], y1 = y + i*diry[dir];
    //             in_range = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
    //             if (in_range && s_cur[x1][y1][0] == p_color)
    //                 s_cur[x][y][dir]++;
    //             else
    //                 break;
    //         }
    //     }
    // };
    // auto evaluate_strength = [&](int x, int y) {
    //     top_cur[y][1] = 0; // reset
    //     for (int dir = 1; dir < 8; dir++) { // consider one dir only
    //         int x1 = x + dirx[dir], y1 = y + diry[dir];
    //         bool in_range1 = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
    //         if (in_range1) {
    //             if (s_cur[x][y][dir] >= 3) {
    //                 if (s_cur[x1][y1][0] == turn) {
    //                     top_cur[y][1] = 3;
    //                     return;
    //                 } else { top_cur[y][1] = 2; }
    //             } else if (s_cur[x][y][dir] > 0) {
    //                 top_cur[y][1] = max(1, top_cur[y][1]);
    //             }
    //         }
    //     }
    //     for (int dir = 1; dir < 3; dir++) { // for each pair of dirs
    //         int x1 = x + dirx[dir], y1 = y + diry[dir];
    //         int x2 = x + dirx[8-dir], y2 = y + diry[8-dir];
    //         bool in_range1 = 0 <= x1 && x1 < nrow && 0 <= y1 && y1 < ncol;
    //         bool in_range2 = 0 <= x2 && x2 < nrow && 0 <= y2 && y2 < ncol;
    //         if (in_range1 && in_range2) { // both dir available
    //             if (s_cur[x1][y1][0] == s_cur[x2][y2][0]) { // same color
    //                 int strength = s_cur[x][y][dir] + s_cur[x][y][8-dir];
    //                 if (strength >= 3) { // 3 or 2
    //                     if (s_cur[x1][y1][0] == turn) { // 3
    //                         top_cur[y][1] = 3;
    //                         return;
    //                     } else { top_cur[y][1] = 2; }
    //                 // TODO: is this reasonable?
    //                 } else if (strength > 0) {
    //                     top_cur[y][1] = max(1, top_cur[y][1]);
    //                 }
    //             }
    //         }
    //     }
    // };
    // auto mark_moves = [&]() {
    //     for (int y = 0; y < ncol; y++) {
    //         if (top_cur[y][0] == 0) continue;
    //         int x = top_cur[y][0] - 1;
    //         mark_location(s_cur, nrow, ncol, x, y);
    //         evaluate_strength(s_cur, top_cur, turn, nrow, ncol, x, y);
    //     }
    // };
    /*
    auto has_game_ended = [&](MyPoint p) {
        int p_color = s_cur[p.first][p.second];
        bool flag[5] = {true, true, true, true, true};
        // int best = 0;
        for (int i = 1; i < 4; i++) { // check 3 consecutive pieces
            for (int j = 0; j < 5; j++) {
                if (!flag[j]) continue;
                int x = p.first + dirx[j] * i;
                int y = p.second + diry[j] * i;
                bool in_range = 0 <= x && x < nrow && 0 <= y && y < ncol;
                if (in_range && s_cur[x][y] != p_color) flag[j] = false;
            }
        }
        return flag[0] || flag[1] || flag[2] || flag[3] || flag[4]; 
    };
    */
    auto is_draw = [&] {
        for (int i = 0; i < ncol; i++)
            if (top_cur[i] > 0)
                return false;
        return true;
    };

    // simulate until one side wins
    ::srand(time(0));
    bool is_ended = false;
    bool is_drawn = false;
    do {
        if ((is_drawn = is_draw())) break; // check for draw

        // mark_locations in top
        for (int y = 0; y < ncol; y++) {
            if (top_cur[y][0] == 0) continue;
            int x = top_cur[y][0] - 1;
            mark_location(s_cur, nrow, ncol, x, y);
            evaluate_strength(s_cur, top_cur, turn, nrow, ncol, x, y);
        }
        // check if a certain move wins or prevents loss
        int y = -1;
        bool has_forcing_moves = false;
        bool has_good_moves = false;
        for (int i = 0; i < ncol; i++) { // check for forcing moves
            if (top_cur[i][0] == 0) continue;
            if (top_cur[i][1] == 3) {
                is_ended = true;
                has_forcing_moves = true, y = i;
                break;
            } else if (top_cur[i][1] == 2) {
                has_forcing_moves = true, y = i;
            }
        }
        if (!has_forcing_moves) {
            for (int i = 0; i < ncol; i++) { // check for good moves
                if (top_cur[i][0] > 0 && top_cur[i][1] > 0) {
                    has_good_moves = true;
                    break;
                }
            }
        }
        if (!has_forcing_moves) { // make a random move among good moves
            int level = has_good_moves ? 1 : 0;
            while (true) { // generate legal move
                y = rand() % ncol;
                if (top_cur[y][0] > 0 && top_cur[y][1] == level)
                    break;
            }
        }
        top_cur[y][0]--;
        s_cur[top_cur[y][0]][y][0] = turn; // make move
        turn = flip(turn);
    } while (!is_ended);

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