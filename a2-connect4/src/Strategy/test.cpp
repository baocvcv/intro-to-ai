#include <iostream>
#include "Strategy.h"
#include "Point.h"

int s[100] = {
    0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,  0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,  0, 1, 0, 0, 0,
    0, 0, 0, 0, 2,  0, 1, 0, 0, 0,
    0, 1, 0, 0, 1,  0, 2, 0, 0, 0,
    1, 1, 2, 1, 2,  2, 1, 0, 0, 0,
};

int top[10];

int main() {
    int M = 10;
    int N = 10;

    for (int j = 0; j < N; j++) {
        int i;
        for (i = 0; i < N && s[i*10+j] == 0; i++) ;
        top[j] = i;
    }

    int x = 7;
    int y = 6;

    // no moves on board
    Point* p = getPoint(M, N, top, s, x, y, 0, 0);
    std::cout << p->x << ' ' << p->y << '\n';

    return 0;
}