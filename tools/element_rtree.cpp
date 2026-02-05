//
// element_rtree.cpp
//
// This is a direct port of the C version of the RTree test program.
//

#include "RTree.h"

#if STELLARATOR_MODEL
constexpr unsigned int ND = 3;
#else
constexpr unsigned int ND = 2;
#endif

extern "C"
{ // prevent name mangling
    using namespace std;

    typedef int ValueType;
    typedef RTree<ValueType, double, ND, double> MyTree;
    // Persistent tree
    static MyTree ElementTree;

    void PopulateTree(int n_elms, int *sizes, double **min, double **max)
    {
        ElementTree.RemoveAll();

        for (int i = 0; i < n_elms; i++) {
            for (int j = 0; j < sizes[i]; j++) {
                double split_min[ND], split_max[ND];
                
                // Copy coords to local variables
                for(unsigned int d=0; d<ND; ++d) {
                    split_min[d] = min[i][j*ND+d];
                    split_max[d] = max[i][j*ND+d];
                }

                // Insert first box
                ElementTree.Insert(split_min, split_max, i + 1);
            }
        }
    }


    // Return the number of elements in a rectangle
    int NumElementsInRect(double *min, double *max)
    {
        return ElementTree.Search(min, max, NULL, NULL);
    }

    // Return element indices of elements contained within the rectangle in element_tree
    // i_elm must be allocated by the caller to size at least nelm.
    int ElementsInRect(double *min, double *max, int *ielm)
    {
        return ElementTree.Search(min, max, NULL, ielm);
    }
}
