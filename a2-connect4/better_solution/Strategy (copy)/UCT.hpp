#ifndef UCT_h
#define UCT_h

#include <utility>
#include <ctime>
#include <memory>

#include "Config.hpp"
#include "Point.h"

class Board;
class TreeNode;

class UCT {
public:

	static double defaultPolicy(TreeNode *nowNode);
	static TreeNode *treePolicy(TreeNode *nowNode);

	static Point* UCTSearch(const int* boardStart, const int* topStart, int _M, int _N, int _noX, int _noY);

	~UCT();
	
	static Board board;
	static int top_cur[MAX_N];
	static int M, N, noX, noY;

	enum class MoveType {
		AVERAGE,
		FORCING,
		LOSING
	};

private:
	UCT() {}

	using MoveResult = std::pair<MoveType, Point>;

	static TreeNode *root;
	static clock_t time0;

	static const int dirx[7], diry[7];

	static const Point NO_MOVE;
	static const MoveResult LOSING;
	static const MoveResult AVERAGE;

	static const MoveResult getOptimalMove(Board &board_cur, int last_player, Point last_move);
	static int getGain(Board& board_cur, int remaining_moves, int player);
};
#endif /* UCT_h */ 
