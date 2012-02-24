/*******************************************************
 * Copyright CNRS 
 * David Coeurjolly
 * david.coeurjolly@liris.cnrs.fr
 * 
 * 
 * This software is a computer program whose purpose is to [describe
 * functionalities and technical features of your software].
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
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <GateDMapVol.h>
#include <GateDMaplongvol.h>
#include <GateDMapoperators.h>

#ifndef __SEDT_H
#define __SEDT_H

using namespace std;

long F(int x, int i, long gi2);
long Sep(int i, int u, long gi2, long gu2);
void phaseSaitoX(const Vol &V, Longvol &sdt_x);
void phaseSaitoY(const Vol &V,Longvol &sdt_x, Longvol &sdt_xy);
void phaseSaitoZ(const Vol &V,Longvol &sdt_xy, Longvol &sdt_xyz);

#endif
