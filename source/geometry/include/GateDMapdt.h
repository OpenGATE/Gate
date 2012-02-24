
/*******************************************************
 * Copyright CNRS
 * David Coeurjolly
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

#ifndef __DT_H
#define __DT_H

//#define _MULTITHREAD 1

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <GateDMapVol.h>
#include <GateDMaplongvol.h>

#ifdef _MULTITHREAD
#include <pthread.h>
#endif



namespace MyVersion{

	//Date Version Types
	static const char DATE[] = "01";
	static const char MONTH[] = "09";
	static const char YEAR[] = "2008";

	//Software Status
	static const char STATUS[] = "Alpha";
	static const char STATUS_SHORT[] = "a";

	//Standard Version Type
	static const long MAJOR = 0;
	static const long MINOR = 5;
	static const long BUILD = 3;
	static const long REVISION = 0;

	//Miscellaneous Version Types
	static const char FULLVERSION_STRING[] = "0.5.3";


}
using namespace std;


///
/// @brief  SEDT computation of the input Vol structure using the Saito's algorithm
/// @param  input the inut vol structure
/// @param  output SEDT result
/// @param  isMultiregion if false, background voxels are zero valued voxels. Otherwise, each value defines an region
/// @return true in case of success
///
bool computeSEDT(const Vol &input, Longvol &output, const bool isMultiregion=false, const bool isToric = false, unsigned int NbThreads=1);


///
/// @brief REDT Computation from a DT
/// @param input an input DT
/// @param shape the reconstructed shape
/// @param value output shape voxel value (optional)
bool computeREDT(const Longvol &input, Vol &shape, const unsigned char value = 255);


///
/// @brief  RDMA Computation from a longvol file containing the SEDT
/// @param  LV the SEDT
/// @return Discrete volume where non-zero valuses defines radii of maximal balls
///
bool computeRDMA(const Longvol &LV, Longvol &rdma);
///
/// @brief  RDMA Computation from a longvol file containing the SEDT
/// @param  LV the SEDT
/// @return Discrete volume where non-zero valuses defines radii of maximal balls
///
bool computeRDMA(const Longvol &LV, Vol &rdma, const unsigned char value = 255);


///
/// @brief  RDMA Computation from a longvol structure containing the SEDT
/// @param  LV the SEDT
/// @return Discrete volume where non-zero valuses defines radii of maximal balls
/// @return The Power discrete diagram (dx,dy,dz)
///
bool computeDiscretePowerDiagram(const Longvol &V, Longvol &dx, Longvol &dy, Longvol &dz);
bool computeDiscretePowerDiagram(const Longvol &V, Longvol &rdma, Longvol &dx, Longvol &dy, Longvol &dz);

#endif

