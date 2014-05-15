/*******************************************************
 * david.coeurjolly@liris.cnrs.fr
 *
 *
 * This software is a computer program whose purpose is to compute the
 * Euclidean distance transformation, the reverse Euclidean distance
 * transformation and the Discrete Medial Axis of a discrete object.
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 *  * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.

 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
*******************************************************/

#include "GateDMapdt_core.h"
#include <iostream>

//inline operators
#include "GateDMapoperators.ihh"


void lowerEnveloppeComputation(long int *sdt_temp, const unsigned int SIZE, const unsigned int TORIC_SHIFT,unsigned int *s,unsigned int *t, int &q)
{
  unsigned int /*u,*/w;

    q = 0;
    s[0] = 0;
    t[0] = 0;

    //Forward Scan
    for (unsigned int u=1; u < SIZE  + TORIC_SHIFT; u++)
    {
        while ((q >= 0) &&
                (F(t[q],s[q],sdt_temp[s[q]]) >
                 F(t[q],u,sdt_temp[u])))
            q--;

        if (q<0)
        {
            q=0;
            s[0]=u;
        }
        else
        {
            w = 1 + Sep(s[q],
                        u,
                        sdt_temp[s[q]],
                        sdt_temp[u]);

            if (w < SIZE + TORIC_SHIFT)
            {
                q++;
                s[q]=u;
                t[q]=w;
            }
        }
    }
}

void upperEnveloppeComputation(long int *sdt_temp, const unsigned int SIZE, const unsigned int TORIC_SHIFT,unsigned int *s,unsigned int *t, int &q)
{
  unsigned int /*u,*/w;

    q = 0;
    s[0] = 0;
    t[0] = 0;

    //Forward Scan
    for (unsigned int u=1; u < SIZE + TORIC_SHIFT ; u++)
    {
        while ((q >= 0) &&
                (F_inv(t[q],s[q],sdt_temp[s[q]]) <
                 F_inv(t[q],u,sdt_temp[u])))
            q--;

        if (q<0)
        {
            q=0;
            s[0]=u;
        }
        else
        {
            w = 1 + Sep_inv(s[q],
                            u,
                            sdt_temp[s[q]],
                            sdt_temp[u]);

            if (w < SIZE + TORIC_SHIFT)
            {
                q++;
                s[q]=u;
                t[q]=w;
            }
        }
    }

}

