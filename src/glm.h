#pragma once

// http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf

namespace npycrf{
    class GLM{
    private:
        int _coverage;
    public:
        // 論文ではcoverage = 8
        // t以前の何文字から素性ベクトルを作るか
        GLM(int coverage = 8){
            _coverage = coverage;
        }
    };
} // namespace npycrf