/*-------------------------------------------------------

           List Mode Format 
                        
     --  locateRsectorInLaboratory.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of locateRsectorInLaboratory.c:

	Find the 3D coordinates of an event in the scanner coordinates (x,y,z) system
	->locateRsectorInLaboratory - Calculate the rsector (id_t,id_z) position 
	                              in the 3D laboratory system (x,y,z)
-------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lmf.h"

/* locateRsectorInLaboratory - Calculate the rsector (id_t,id_z) position 
   in the 3D laboratory system (x,y,z) */

coordinates locateRsectorInLaboratory(u16 * pcrist,
				      LMF_cch_scannerGeometry * pScanGeo,
				      const ENCODING_HEADER * pEncoH,
				      const EVENT_RECORD * pER,
				      double *first_pIntLengthLayers,
				      double *first_pRdSizeLayers,
				      double
				      angleDefaultUnitToRadConversionFactor,
				      double
				      *first_pSubstructuresNumericalValues)
{

  coordinates centerOfTheFirstRowOfCrystals =
      { 0 }, rsectorInLaboratoryPosition = {
  0};
  generalSubstructureID rsectorID = { 0 };

  int list_index = 0;
  double radialPitch = 0, totalLayersRadialSize = 0, ringRadius =
      0, rotationAngle = 0;
  double *pRdSizeLayers = NULL, *pIntLengthLayers =
      NULL, *pSubstructuresNumericalValues = NULL;

  u16 firstRowOfCrystalsIDs[5] = { 0, 0, 0, 0, 0 };	/*   layer with id_z=0
							   + crystal with id_z=0
							   + submodule with id_z=0
							   + module with id_z=0
							   + rsector with id_z=0 */
  u16 *pFirstRowOfCrystalsIDs = NULL;

  /**** Find the axial and tangential rsectorID (id_t,id_z) ****/
  rsectorID = locateID(pcrist, 4, first_pSubstructuresNumericalValues);

  /**** calculation of the radialPitch ****/
  pSubstructuresNumericalValues = first_pSubstructuresNumericalValues + 15;

  if ((int) pcrist[0] > (int) pSubstructuresNumericalValues[0]) {
    printf(ERROR_GEOMETRY3, (int) pcrist[0],
	   (int) pSubstructuresNumericalValues[0]);
    printf(ERROR5, cchFileName);
    exit(EXIT_FAILURE);
  }

  for (list_index = 0; list_index < (int) pcrist[0]; list_index++) {
    pRdSizeLayers = first_pRdSizeLayers + list_index;
    totalLayersRadialSize = totalLayersRadialSize + (*pRdSizeLayers);
  }

  pIntLengthLayers = first_pIntLengthLayers + (int) pcrist[0];
  radialPitch = totalLayersRadialSize + (*pIntLengthLayers);

  ringRadius = (double) (0.5 * pScanGeo->ringDiameter);

  rotationAngle = (double) ((double) rsectorID.tangential
			    * (pScanGeo->rsectorAzimuthalPitch *
			       angleDefaultUnitToRadConversionFactor));
  /* //                  +(double)((pScanGeo->azimuthalStep*angleDefaultUnitToRadConversionFactor)*(double)pER->gantryAngularPos)); */

  /** x-coordinate calculation of the rsector (id_t,id_z) in the 3D laboratory (x,y,z) system **/
  /*  x'-coordinate calculation in the 3D scanner (x',y',z') system, 
     so that 0x' points to the center of the rsector0 */
  /* =x-coordinate calculation in the 3D laboratory (x,y,z) system, 
     so that 0x points to the center of the first row of crystals */

  rsectorInLaboratoryPosition.radial =
      ((ringRadius + radialPitch) * cos(rotationAngle));

  /** y-coordinate calculation of the rsector (id_t,id_z) in the 3D laboratory (x,y,z) system **/
  /*  y'-coordinate calculation in the 3D scanner (x',y',z') system, 
     so that 0x' points to the center of the rsector0 */
  /* =y-coordinate calculation in the 3D scanner (x,y,z) system, 
     so that 0x points to the center of the first row of crystals */

  rsectorInLaboratoryPosition.tangential =
      ((ringRadius + radialPitch) * sin(rotationAngle));

  /** z-coordinate calculation of the rsector (id_t,id_z) in the 3D laboratory (x,y,z) system**/
  /*  z'-coordinate calculation in the 3D scanner (x',y',z') system, 
     so that 0x' points to the center of the rsector0 */

  rsectorInLaboratoryPosition.axial =
      (double) (rsectorID.axial * pScanGeo->rsectorAxialPitch);

  /*  position of the center of a Rsector in the 3D laboratory (x,y,z) system 
     with 0x pointing to the center of the first row of crystals */
  pFirstRowOfCrystalsIDs = &firstRowOfCrystalsIDs[1];
  centerOfTheFirstRowOfCrystals =
      locateSubstructureInStructure(pFirstRowOfCrystalsIDs, 1,
				    first_pSubstructuresNumericalValues);

  /*  z-coordinate calculation in the laboratory 3D coordinates system (x,y,z), 
     so that 0x points to the center of the first row of crystals */
  rsectorInLaboratoryPosition.axial = rsectorInLaboratoryPosition.axial
      - centerOfTheFirstRowOfCrystals.axial
      + (pScanGeo->axialStep * (double) pER->gantryAxialPos);

  return (rsectorInLaboratoryPosition);
}
