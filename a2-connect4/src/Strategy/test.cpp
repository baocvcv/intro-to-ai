#include <iostream>
#include "Strategy.h"
#include "Point.h"

int main() {
    int M = 10;
    int N = 10;
    int s[100];
    int top[10];
    for (int j = 0; j < 10; j++) {
        top[j] = 10;
        for (int i = 0; i < 10; i++) {
            s[i * 10 + j] = 0;
        }
    }

    // no moves on board
    Point* p = getPoint(M, N, top, s, -1, -1, 5, 5);
    std::cout << p->x << ' ' << p->y << '\n';

    s[9 * 10 + 0] = 1;
    s[9 * 10 + 7] = 2;
    top[0] = 9;
    top[7] = 9;
    p = getPoint(M, N, top, s, 9, 0, 2, 9);
    std::cout << p->x << ' ' << p->y << '\n';

    return 0;
}