#include <cstdlib>
#include <limits>

#include "Node.hpp"
#include "Board.hpp"
#include "UCT.hpp"
#include "Helpers.hpp"

TreeNode** TreeNode::pool = new TreeNode*[MAX_MEMORY_POOL_SIZE];

TreeNode::TreeNode(int x, int y, int player, TreeNode *_parent) {
	set(x, y, player, _parent);
}

size_t TreeNode::usedMemory = 0;

void TreeNode::set(int x, int y, int player, TreeNode *_parent) {
	profit = .0; visNum = 0;
	_isTerminal = Status::UNDETERMINED; // not determined
	parent = _parent; _player = player;
	_x = x; _y = y;
	expandableNumber = 0;
	for (int i = 0; i < UCT::N; i++) {
		if (UCT::top_cur[i] > 0) {
			expandableNodes[expandableNumber++] = i;
		}
		children[i] = nullptr;
	}
}

bool TreeNode::isTerminal()
{
	if (_isTerminal != Status::UNDETERMINED) {
		return _isTerminal == Status::LEAF;
	} else if (expandableNumber == 0 || UCT::board.is_won(1-_player)) {
		_isTerminal = Status::LEAF;
		return true;
	} else {
		_isTerminal = Status::BRANCH;
		return false;
	}
}

void TreeNode::backPropagation(double delta)
{
	TreeNode *nowNode = this;
	while (nowNode != nullptr)
	{
		nowNode->profit += delta;
		nowNode->visNum++;
		nowNode = nowNode->parent;
	}
}

TreeNode* TreeNode::expand() {
	int index = rand() % expandableNumber;
	int newY = expandableNodes[index], newX = --UCT::top_cur[newY];
	UCT::board.make_move(newX, newY, _player);
	if (newX - 1 == UCT::noX && newY == UCT::noY) {
		UCT::top_cur[newY]--;
	}

	if (usedMemory >= MAX_MEMORY_POOL_SIZE) {
		children[newY] = new TreeNode(newX, newY, 1-_player, this);
	} else if (pool[usedMemory] == nullptr) {
		children[newY] = pool[usedMemory] = new TreeNode(newX, newY, 1-_player, this);
		usedMemory++;
	} else {
		(children[newY] = pool[usedMemory])->set(newX, newY, 1-_player, this);
		usedMemory++;
	}

	std::swap(expandableNodes[index], expandableNodes[--expandableNumber]);
	return children[newY];
}

TreeNode* TreeNode::bestChild(bool placeNewChess) {
	TreeNode *bestNode = children[0];
	double maxProfit = -std::numeric_limits<double>::max();
	int bestY = 0;
	for (int i = 0; i < UCT::N; i++) {
		if (children[i] == nullptr) {
			continue;
		}
		auto cur_score = children[i]->calc_score(visNum);
		if (cur_score > maxProfit - eps) {
			maxProfit = cur_score;
			bestNode = children[i];
			bestY = i;
		}
	}
	if (!placeNewChess) {
	} else {
		UCT::board.make_move(--UCT::top_cur[bestY], bestY, _player);
		if (bestY == UCT::noY && UCT::top_cur[bestY] - 1 == UCT::noX) {
			UCT::top_cur[bestY]--;
		}
	}

	return bestNode;
}
