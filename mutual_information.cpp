#include <iostream>
#include "table.h"

int main(int argc, char* argv[]) {
    bool isPrivate = std::atoi(argv[1]);
    std::string dataset_path = "adult";
    int col1 = 9;
    int col2 = 14;

    table t(dataset_path, false);  // false = not using func1 mode

    std::vector<int> cols = {col1, col2};
    std::vector<int> lvls = {0, 0};  // no generalization

    std::vector<double> counts = t.getCounts(cols, lvls);
    std::vector<int> widths = t.getWidth(cols, lvls);

    vector<double> result;
    if (isPrivate) {
        result = t.getF(counts, widths);
    } else {
        result = t.getI(counts, widths);
    }
    std::cout << result[0] << std::endl;

    return 0;
}

