/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */


#include "pcg_basic.h"
#include <stdio.h>

// pcg32_srandom_r(rng, initstate, initseq):
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)

void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}

// pcg32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * PCG_DEFAULT_MULTIPLIER_64 + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// pcg32_random_double_r(rng)
//      Generate a 32-bit floating point number between 0 and 1

double pcg32_random_double_r(pcg32_random_t* rng)
{
    return ldexp(pcg32_random_r(rng), -32);
}


/* Multi-step advance functions (jump-ahead, jump-back)
 *
 * The method used here is based on Brown, "Random Number Generation
 * with Arbitrary Stride,", Transactions of the American Nuclear
 * Society (Nov. 1994).  The algorithm is very similar to fast
 * exponentiation.
 *
 * Even though delta is an unsigned integer, we can pass a
 * signed integer to go backwards, it just goes "the long way round".
 */

uint64_t pcg_advance_lcg_64(uint64_t state, uint64_t delta, uint64_t cur_mult,
                            uint64_t cur_plus)
{
    uint64_t acc_mult = 1u;
    uint64_t acc_plus = 0u;
    while (delta > 0) {
        if (delta & 1) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        delta /= 2;
    }
    return acc_mult * state + acc_plus;
}

// pcg32_advance_r
//      Advance by delta steps in one go

void pcg32_advance_r(pcg32_random_t* rng, uint64_t delta)
{
    rng->state = pcg_advance_lcg_64(rng->state, delta,
                                    PCG_DEFAULT_MULTIPLIER_64, rng->inc);
}
