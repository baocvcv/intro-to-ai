#ifndef TREENODE_H
#define TREENODE_H

#include <cstddef>
#include <cstdio>
#include "Config.hpp"
#include "Helpers.hpp"
#include "UCT.hpp"

class TreeNode
{
	friend class UCT;

	enum class Status {
		UNDETERMINED,
		BRANCH,
		LEAF,
	};

public:
	TreeNode(int x, int y, int player, TreeNode *_parent);
	~TreeNode() {
		if (parent == nullptr) fprintf(stderr, "In node destructor\n");
		for (int i = 0; i < UCT::N; i++) {
			if (children[i] != nullptr) {
				delete children[i];
			}
		}
	}
	bool isTerminal();
	void set(int x, int y, int player, TreeNode *_parent);
	int x() { return _x; }
	int y() { return _y; }
	void backPropagation(double delta);
	double calc_score(int N) {
		double profit = this->profit * (_player == PLAYER_ME ? 1 : -1);
		return profit / visNum + COEFFICIENT * fastSqrt(2 * fastLog(N) / visNum);
	}
	TreeNode* bestChild(bool placeNewChess = true);
	TreeNode* expand();

	static size_t usedMemory;

private:
	int _player;
	int _x, _y;
	int visNum;
	double profit;
	TreeNode* parent;
	TreeNode* children[MAX_N];
	int expandableNumber;
	int expandableNodes[MAX_N];
	Status _isTerminal;
	static TreeNode **pool;
};

#endif