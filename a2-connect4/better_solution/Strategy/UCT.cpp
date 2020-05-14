#include <ctime>
#include <cstring>
#include <cstdlib>
#include <memory>

#include "UCT.hpp"
#include "Node.hpp"
#include "Board.hpp"
#include "Strategy.h"

TreeNode* UCT::root = nullptr;
int UCT::top_cur[MAX_N] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
Board UCT::board;
int UCT::M = 0, UCT::N = 0, UCT::noX = 0, UCT::noY = 0;
const int UCT::dirx[7] = { 0,-1, 1, 1,-1, 1, 0};
const int UCT::diry[7] = {-1,-1,-1, 0, 1, 1, 1};

const Point UCT::NO_MOVE({ -1, -1 });
const UCT::MoveResult UCT::LOSING = { UCT::MoveType::LOSING, NO_MOVE };
const UCT::MoveResult UCT::AVERAGE = { UCT::MoveType::AVERAGE, NO_MOVE };

UCT::~UCT() {
	delete root;
}

TreeNode* UCT::treePolicy(TreeNode *node) {
	while (!node->isTerminal()) {
		if (node->expandableNumber <= 0) {
			node = node->bestChild();
		}
		else {
			return node->expand();
		}
	}
	return node;
}

const UCT::MoveResult UCT::getOptimalMove(Board &board_cur, int last_player, Point last_move) {
	// downwards
	int down;
    for (down = last_move.x + 1; down < M; down++) {
        if (!board_cur.check_loc(down, last_move.y, last_player)) break;
    }
    if (last_move.x - down == 3 && (last_move.y != noY || last_move.x-1 != noX)) {
		return { MoveType::FORCING, { last_move.x-1, last_move.y } };
	}

    // check other dirs
    for (int dir = 0; dir < 3; dir++) {
        int x1, y1;
        for (x1 = last_move.x+dirx[dir], y1 = last_move.y+diry[dir];
             0 <= x1 && x1 < M && 0 <= y1 && y1 < N;
             x1 += dirx[dir], y1 += diry[dir])
        {
            if (!board_cur.check_loc(x1, y1, last_player)) break;
        }
        int x2, y2;
        for (x2 = last_move.x+dirx[8-dir], y2 = last_move.y+diry[6-dir];
             0 <= x2 && x2 < M && 0 <= y2 && y2 < N;
             x2 += dirx[6-dir], y2 += diry[6-dir])
        {
            if (!board_cur.check_loc(x2, y2, last_player)) break;
        }

        if (y2 - y1 >= 4) { // 3 pieces in between
            bool empty1 = (0 <= x1 && x1 < M && 0 <= y1 && y1 < N) && top_cur[y1] == x1 + 1;
            bool empty2 = (0 <= x2 && x2 < M && 0 <= y2 && y2 < N) && top_cur[y2] == x2 + 1;
            if (empty1 && empty2)
                return { MoveType::LOSING, NO_MOVE };
            else if (empty1)
				return { MoveType::FORCING,{ x1, y1 } };
            else if (empty2)
				return { MoveType::FORCING,{ x2, y2 } };
        }
    }

	return AVERAGE;
}


inline int UCT::getGain(Board& board_cur, int remaining_moves, int player) {
	if (board_cur.is_won(1-player)) {
		return (player == PLAYER_ME ? PROFIT_OPPONENT_WIN : PROFIT_I_WIN);
	}
	return (remaining_moves == 0 ? PROFIT_TIE : UNTERMINAL_STATE);
}

double UCT::defaultPolicy(TreeNode *nowNode) {

	int player = nowNode->_player;
	Point last_move({ nowNode->_x, nowNode->_y });

	int remaining_moves = 0;
	for (int i = 0; i < N; i++) {
		remaining_moves += top_cur[i];
	}
	auto& board_cur = board;

	auto gain = getGain(board_cur, remaining_moves, player);
	while (gain == UNTERMINAL_STATE) {
		auto result = getOptimalMove(board_cur, 1-player, last_move);
		switch (result.first) {
		case UCT::MoveType::LOSING:
			return player == PLAYER_OPPONENT ? PROFIT_I_WIN : PROFIT_OPPONENT_WIN;
		case UCT::MoveType::FORCING:
			last_move = result.second;
			board_cur.make_move(last_move.x, last_move.y, player);
			--top_cur[last_move.y];
			break;
		case UCT::MoveType::AVERAGE:
			int y = rand() % N;
			while (top_cur[y] == 0) {
				y = rand() % N;
			}
			last_move = { --top_cur[y], y };
			board_cur.make_move(last_move.x, last_move.y, player);
			break;
		}

		if (last_move.x - 1 == noX && last_move.y == noY) {
			top_cur[last_move.y]--;
			remaining_moves -= 2;
		} else {
			remaining_moves--;
		}
		player = 1 - player;
		gain = getGain(board, remaining_moves, player);
	}
	return (double)gain;
}

Point* UCT::UCTSearch(const int* boardStart, const int* topStart, int _M, int _N, int _noX, int _noY)
{
	auto time0 = clock();
	M = _M; N = _N; noX = _noX; noY = _noY;

	Board board_start(M, N, boardStart);

	for (int i = 0; i < N; i++) top_cur[i] = topStart[i];
	root = new TreeNode(-1, -1, PLAYER_OPPONENT, nullptr);
	TreeNode::usedMemory = 0;
	int times = 0;

	srand(time(0));

	auto timed = 1e6 * (clock() - time0) / CLOCKS_PER_SEC;
	while (timed < TIME_LIMIT_MICROSECOND) {
		board = board_start;
		for (int i = 0; i < N; i++) top_cur[i] = topStart[i];
		auto nowNode = treePolicy(root);
		auto delta = defaultPolicy(nowNode);
		nowNode->backPropagation(delta);
		times++;
		timed = 1e6 * (clock() - time0) / CLOCKS_PER_SEC;
	}

	auto best = root->bestChild(false);

#ifdef _DEBUG
	fprintf(stderr, "Searched %d times, taking %lf ms\n", times, timed/1000);
#endif

	return new Point(best->x(), best->y());
}