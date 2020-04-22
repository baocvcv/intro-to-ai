#include <iostream>
#include <unistd.h>
#include <memory>
#include <ctime>

#include "Point.h"
#include "Strategy.h"
#include "Node.h"

using namespace std;

const int MAX_TREE_SIZE = 800000;
// const int MAX_BOARD_SIZE = 12;

int M, N;
// int **board;

int tree_policy(unique_ptr<Node[]>& nodes, int& node_cnt);
void backup(unique_ptr<Node[]>& nodes, int gain, int idx);
// Returns the local index of the best child of nodes[idx]
int best_child(unique_ptr<Node[]>& nodes, int idx);

/*
	策略函数接口,该函数被对抗平台调用,每次传入当前状态,要求输出你的落子点,该落子点必须是一个符合游戏规则的落子点,不然对抗平台会直接认为你的程序有误
	
	input:
		为了防止对对抗平台维护的数据造成更改，所有传入的参数均为const属性
		M, N : 棋盘大小 M - 行数 N - 列数 均从0开始计， 左上角为坐标原点，行用x标记，列用y标记
		top : 当前棋盘每一列列顶的实际位置. e.g. 第i列为空,则_top[i] == M, 第i列已满,则_top[i] == 0
		_board : 棋盘的一维数组表示, 为了方便使用，在该函数刚开始处，我们已经将其转化为了二维数组board
				你只需直接使用board即可，左上角为坐标原点，数组从[0][0]开始计(不是[1][1])
				board[x][y]表示第x行、第y列的点(从0开始计)
				board[x][y] == 0/1/2 分别对应(x,y)处 无落子/有用户的子/有程序的子,不可落子点处的值也为0
		lastX, lastY : 对方上一次落子的位置, 你可能不需要该参数，也可能需要的不仅仅是对方一步的
				落子位置，这时你可以在自己的程序中记录对方连续多步的落子位置，这完全取决于你自己的策略
		noX, noY : 棋盘上的不可落子点(注:涫嫡饫锔?龅膖op已经替你处理了不可落子点，也就是说如果某一步
				所落的子的上面恰是不可落子点，那么UI工程中的代码就已经将该列的top值又进行了一次减一操作，
				所以在你的代码中也可以根本不使用noX和noY这两个参数，完全认为top数组就是当前每列的顶部即可,
				当然如果你想使用lastX,lastY参数，有可能就要同时考虑noX和noY了)
		以上参数实际上包含了当前状态(M N _top _board)以及历史信息(lastX lastY),你要做的就是在这些信息下给出尽可能明智的落子点
	output:
		你的落子点Point
*/
extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
						   const int lastX, const int lastY, const int noX, const int noY)
{
	clock_t time0 = clock();

	int board[MAX_BOARD_SIZE][MAX_BOARD_SIZE];
	::M = M, ::N = N;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			board[i][j] = _board[i * N + j];

	#ifdef DEBUG
	cerr << "Current board:\n";
	for (int i = 0; i < M * N; i++)
		cerr << _board[i] << ((i+1) % N == 0 ? '\n' : ' ');
	cerr << "Top: ";
	for (int i = 0; i < N; i++)
		cerr << top[i] << ' ';
	cerr << endl;
	#endif

	// Node* nodes = new Node[MAX_TREE_SIZE];
	unique_ptr<Node[]> nodes(new Node[MAX_TREE_SIZE]);
	int node_cnt = 1;
	// init root node
	nodes[0].init(-1, board, M, N, MyPoint(lastX, lastY), top, 1);
	backup(nodes, nodes[0].defaultPolicy(), 0);
	if (nodes[0].is_leaf()) {
		int y = nodes[0].the_move;
		return new Point(top[y]-1, y);
	}

	#ifndef DEBUG
	clock_t time1 = clock();
	int time_d = 1000 * (time1 - time0) / CLOCKS_PER_SEC;
	while (time_d < 2800) {
	#else
	int time_d = 1;
	while (time_d < 10000) {
	#endif
		// select new node
		int child_idx = tree_policy(nodes, node_cnt);
		// simulate & backup
		// for (int i = 0; i < 10; i++) {
			int result;
			if (child_idx == -1) result = 2;
			else result = nodes[child_idx].defaultPolicy();
			backup(nodes, result, child_idx);
		// }

		#ifndef DEBUG
		time_d = 1000 * (time1 - time0) / CLOCKS_PER_SEC;
		#else
		time_d++;
		#endif
	}
	int y = best_child(nodes, 0);
	#ifdef DEBUG
	int n = nodes[0].N;
	for (int i = 0; i < N; i++) {
		if (nodes[0].children[i] > 0) {
			cerr << '(' << nodes[nodes[0].children[i]].Q << ',';
			cerr << nodes[nodes[0].children[i]].N << ",";
			cerr << nodes[nodes[0].children[i]].calc_value(n) << ") ";
		}
	}
	cerr <<  endl;
	cerr << "Move: " << '(' << top[y]-1 << ',' << y << ") ";
	cerr << "Used " << time_d << "ms" << endl << endl;
	#endif
	
	//clearArray(M, N);
	return new Point(top[y]-1, y);
}

/*
	treePolicy + expand
*/
int tree_policy(unique_ptr<Node[]>& nodes, int& node_cnt) {
	int idx = 0;
	while (!nodes[idx].is_leaf()) {
		int expandable = nodes[idx].expandable();
		if (expandable== -2 || expandable >= 0) // have children to expand
			if (node_cnt < MAX_TREE_SIZE) {
				nodes[idx].expand(idx, node_cnt, nodes[node_cnt]);
				return node_cnt++;
			} else {
				cout << "Not enough nodes. Abort" << endl;
				abort();
			}
		else if (expandable == -3) // full
			idx = nodes[idx].children[best_child(nodes, idx)];
		else if (expandable == -1) // winnable by next move
			return -1;
	}
	return idx;
}

void backup(unique_ptr<Node[]>& nodes, int gain, int idx) {
	while (idx != -1) {
		nodes[idx].update(gain);
		gain = 2 - gain;
		idx = nodes[idx].parent;
	}
}

int best_child(unique_ptr<Node[]>& nodes, int idx) {
	if (nodes[idx].is_leaf()) return nodes[idx].the_move;

	// calc best child
	int best_child = -1;
	double best_val = .0;
	int* children = nodes[idx].children;
	for (int i = 0; i < N; i++) {
		if (children[i] > 0) {
			double val = nodes[children[i]].calc_value(nodes[idx].N);
			if (val > best_val) {
				best_val = val;
				best_child = i;
			}
		}
	}
	return best_child;
}

/*
	getPoint函数返回的Point指针是在本so模块中声明的，为避免产生堆错误，应在外部调用本so中的
	函数来释放空间，而不应该在外部直接delete
*/
extern "C" void clearPoint(Point *p)
{
	delete p;
	return;
}

/*
	清除top和board数组
*/
void clearArray(int M, int N, int** board)
{
	for (int i = 0; i < M; i++)
	{
		delete[] board[i];
	}
	delete[] board;
}

/*
	添加你自己的辅助函数，你可以声明自己的类、函数，添加新的.h .cpp文件来辅助实现你的想法
*/
