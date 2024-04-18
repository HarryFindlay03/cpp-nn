/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 27/04/2024
 * FILE LAST UPDATED: 09/05/2024
 * 
 * DESCRIPTION: Class definition and implementation for random distributions.
*/

#ifndef RAND_HELPER_H
#define RAND_HELPER_H

#include <algorithm>
#include <random>

#define RAND_SEED 12

class rand_helper
{
    std::mt19937 gn;
    std::default_random_engine rng;

public:
    rand_helper()
    {
        std::random_device rd;

        std::mt19937 generator(rd());
        gn = generator;

        /* seeding */
        gn.seed(RAND_SEED);
        srand(RAND_SEED);

        return;
    };

    ~rand_helper(){};

    int random_int_range(const int range_from, const int range_to)
    {
        std::uniform_int_distribution<int> distr(range_from, range_to);
        return distr(gn);
    };

    double random_double_range(const double range_from, const double range_to)
    {
        std::uniform_real_distribution<double> distr(range_from, range_to);
        return distr(gn);
    }

    double normal_distribution(const double mean, const double sd)
    {
        std::normal_distribution<double> distr(mean, sd);
        return distr(gn);
    }
};

#endif /* RAND_HELPER_H */