#ifndef BOARD_HPP_
#define BOARD_HPP_

#include "BitSet.hpp"
#include "Config.hpp"

#include <cstdint>
#include <cstdio>

class Board {
public:
    Board(int _M = 0, int _N = 0) :
        nrow(_M), ncol(_N) {}

    Board(int _M, int _N, const int* _board) :
            Board(_M, _N)
    {
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                int _color = _board[i*ncol+j];
                if (_color > 0)
                    BitSet::set(boards[_color-1], i + j * 16);
            }
        }
    }

    // place a piece
    void place(int x, int y, int color) { BitSet::set(boards[color-1], x + 16 * y); }

    // get location info
    bool piece_at(int x, int y, int color) const {
        return BitSet::test(boards[color-1], x + 16 * y);
    }

    // test if game is won
    bool is_won(int side) const {
        auto& board = boards[side-1];
        auto vert = BitSet::andWith(board, BitSet::andWith(BitSet::rightShift(board, 1), BitSet::andWith(BitSet::rightShift(board, 2), BitSet::rightShift(board, 3))));
        auto horWithi = BitSet::andWith(board, BitSet::andWith(BitSet::rightShift(board, 16), BitSet::andWith(BitSet::rightShift(board, 32), BitSet::rightShift(board, 48))));
        auto bslash = BitSet::andWith(board, BitSet::andWith(BitSet::rightShift(board, 15), BitSet::andWith(BitSet::rightShift(board, 30), BitSet::rightShift(board, 45))));
        auto slash = BitSet::andWith(board, BitSet::andWith(BitSet::rightShift(board, 17), BitSet::andWith(BitSet::rightShift(board, 34), BitSet::rightShift(board, 51))));
        auto final = BitSet::orWith(horWithi, BitSet::orWith(vert, BitSet::orWith(bslash, slash)));
        return BitSet::notZero(final);
    }

    // print board
    void print() const {
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                if (piece_at(i, j, 1)) fprintf(stderr, "1 ");
                else if (piece_at(i, j, 2)) fprintf(stderr, "2 ");
                else fprintf(stderr, "- ");
            }
            fprintf(stderr, "\n");
        }
    }

private:
    // boards[0] is my board, boards[1] is opponents board
    BitSet::bits boards[2] = { { 0, 0, 0 }, { 0, 0, 0 }};
    int nrow, ncol;
};


#endif