/*******************************************************
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
/** operators : Basic arithmetic operation using INFTY numbers
 * 
 * David Coeurjolly (david.coeurjolly@liris.cnrs.fr) - Sept. 2004
 *
**/

#include "GateDMapoperators.h"



/////////Basic functions to handle operations with INFTY

/** 
 **************************************************
 * @b sum
 * @param a Long number with INFTY
 * @param b Long number with INFTY
 * @return The sum of a and b handling INFTY
 **************************************************/
 long sum(long a, long b) 
{
  if ((a==INFTY) || (b==INFTY))     
    return INFTY;    
  else 
    return a+b;
}

/** 
 **************************************************
 * @b prod
 * @param a Long number with INFTY
 * @param b Long number with INFTY
 * @return The product of a and b handling INFTY
 **************************************************/
 long prod(long a, long b) 
{
  if ((a==INFTY) || (b==INFTY)) 
    return INFTY;  
  else 
    return a*b;
}
/** 
 **************************************************
 * @b opp
 * @param a Long number with INFTY
 * @return The opposite of a  handling INFTY
 **************************************************/
 long opp (long a) {
  if (a == INFTY) {
    return INFTY;
  }
  else {
    return -a;
  }
}

/** 
 **************************************************
 * @b intdivint
 * @param divid Long number with INFTY
 * @param divis Long number with INFTY
 * @return The division (integer) of divid out of divis handling INFTY
 **************************************************/
 long intdivint (long divid, long divis) {
  if (divis == 0) 
    return  INFTY;
  if (divid == INFTY) 
    return  INFTY;
  else 
    return  divid / divis;
}
//////////

/**
 =================================================
 * @file   operators.cc
 * @author David COEURJOLLY <David Coeurjolly <dcoeurjo@liris.cnrs.fr>>
 * @date   Thu Sep 30 09:22:19 2004
 * 
 * @brief  Basic implementation of arithmetical operations using INFTY numbers.
 * 
 * 
 =================================================*/
