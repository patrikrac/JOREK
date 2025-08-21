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

    void PopulateTree(int n, int n_plane, double *min, double *max)
    {
        ElementTree.RemoveAll();
        for (unsigned int i = 0; i < n * n_plane; i++)
            ElementTree.Insert(min + i * ND, max + i * ND, (i / n_plane) + 1); // store element number (1-based)
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
