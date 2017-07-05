#pragma once
#include <vector>
#include <unordered_set>
#include <string>
using namespace std;

// Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models
// http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf

namespace npycrf{
    class GLM{
    private:
        int _coverage;
        unordered_set<wstring> _word_set;
    public:
        GLM(){
            _coverage = 0;
        }
        // 論文ではcoverage = 8
        // t以前の何文字から素性ベクトルを作るか
        GLM(int coverage){
            _coverage = coverage;
        }
        void add_words(vector<wstring> &words){

        }
    };
} // namespace npycrf