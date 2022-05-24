/*
 * author: Laurens Devos
 * Copyright BDAP team, DO NOT REDISTRIBUTE
 *
 *******************************************************************************
 *                                                                             *
 *                     DO NOT CHANGE SIGNATURES OF METHODS!                    *
 *             DO NOT CHANGE METHODS IMPLEMENTED IN THIS HEADER!               *
 *     Sections which require modifications indicated with 'TODO' comments     *
 *                                                                             *
 *******************************************************************************
 */

#include "prod_quan_nn.hpp"
#include <limits>
#include <chrono>
#include <cmath>

namespace bdap {

    // Constructor, modify if necessary for auxiliary structures
    ProdQuanNN::ProdQuanNN(std::vector<Partition>&& partitions)
        : partitions_(std::move(partitions))
    {}

    void
    ProdQuanNN::initialize_method()
    {
        //std::cout << "Construct auxiliary structures here" << std::endl;
    }

    void
    ProdQuanNN::compute_nearest_neighbors(
                const pydata<float>& examples,
                int nneighbors,
                pydata<int>& out_index,
                pydata<float>& out_distance) const
    { 
        // iterate over all examples
        for (unsigned long i = 0; i < examples.nrows; i++){
            std::vector<float> trainDists(partitions_[0].labels.nrows);
            // iterate over all partitions
            for (unsigned long j = 0; j < partitions_.size(); j++){
                // iterate over all centroids
                const pydata<float> *centroids = &partitions_[j].centroids;
                std::vector<float> lookup;
                for (unsigned long k = 0; k < centroids->nrows; k++){
                    // iterate over all features of that partition and calculate distance to that centroid
                    float distanceToCentroid = 0.0;
                    int centroidInd = 0;
                    for (int l = partitions_[j].feat_begin; l < partitions_[j].feat_end; l++){
                        distanceToCentroid += std::pow((centroids->get_elem(k, centroidInd) - examples.get_elem(i, l)), 2);

                        if (i == 0){
                            out_distance.set_elem(k, centroidInd, std::pow((centroids->get_elem(k, centroidInd) - examples.get_elem(i, l)), 2));
                        }
                        centroidInd++;
                    }
                    // add distance to lookup table
                    lookup.push_back(distanceToCentroid);
                }

                // add distances of the respective label in lookup table
                for (unsigned long m = 0; m < partitions_[j].labels.nrows; m++){
                    trainDists[m] += lookup[partitions_[j].labels[m]];
                }
            }
            // now traindists is filled with all the distances from each train example, now only the closest neighbours need to be found

            for (int j = 0; j < nneighbors; j++){
                int ind = std::min_element(trainDists.begin(), trainDists.end()) - trainDists.begin();
                out_index.set_elem(i, j, std::move(ind));
                out_distance.set_elem(i, j, std::move(trainDists[ind]));
                trainDists.erase(trainDists.begin() + ind);
            }
        }
        //std::cout << "Compute the nearest neighbors for the "
        //    << examples.nrows
        //    << " given examples." << std::endl

        //    << "The examples are given in C-style row-major order, that is," << std::endl
        //    << "the values of a row are consecutive in memory." << std::endl

        //    << "The 5th example can be fetched as follows:" << std::endl;

        //float const *ptr = examples.ptr(5, 0);
        //std::cout << '[';
        //for (size_t i = 0; i < examples.ncols; ++i) {
        //    if (i>0) std::cout << ",";
        //    if (i>0 && i%5==0) std::cout << std::endl << ' ';
        //    printf("%11f", ptr[i]);
        //}
        //std::cout << " ]" << std::endl;
    }
} // namespace bdap
