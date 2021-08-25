/*-------------------------------------------------------

           List Mode Format 
                        
     --  locateEventInLaboratory.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of locateEventInLaboratory.c:
	Find the 3D coordinates of an event in the laboratory coordinates (x,y,z) system
	->locateEventInLaboratory - Calculate events 3D coordinates 
	                            in the 3D laboratory (x,y,z) system.
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lmf.h"

/* locateEventInLaboratory - Calculation of events 3D coordinates 
   in the 3D laboratory (x,y,z) system */

static int initializationDone = FALSE;
static double *first_pIntLengthLayers = NULL, *first_pRdSizeLayers = NULL;

calculOfEventPosition locateEventInLaboratory(const ENCODING_HEADER *
					      pEncoH,
					      const EVENT_RECORD * pER,
					      int indexID)
{
  u16 *pcrist = NULL;
  static LMF_cch_scannerGeometry myScannerGeometry = { 0 };
  static LMF_cch_scannerGeometry *pScanGeo = &myScannerGeometry;
  /* declares pScanGeo to be of pointer to the (struct) LMF_cch_scannerGeometry */

  calculOfEventPosition resultOfCalculOfEventPosition;
  generalSubstructureID rsectorID = { 0 };

  static double *first_pSubstructuresNumericalValues = NULL;
  static double angleDefaultUnitToRadConversionFactor = 0, rotationAngle =
      0;
  double eventPosX = 0, eventPosY = 0, eventPosZ = 0, angularRsectorPos =
      0;
  int rdNbOfLayers = 0, cch_index = 0;

  /**** debuild the ID ****/
  pcrist = demakeid(pER->crystalIDs[indexID], pEncoH);

  if (initializationDone == FALSE) {
    /**** Fill in the structure scannerGeometry ****/
    plist_cch = first_cch_list;

    if (fillInStructScannerGeometry(cch_index, pScanGeo) == 1) {
      printf(ERROR_GEOMETRY1);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }

    /**** Test if the scanner geometry is cylindrical or not ****/
    if (pScanGeo->geometricalDesignType != 1) {
      printf(ERROR_GEOMETRY2);
      exit(EXIT_FAILURE);
    }

    /* Initialize all parameters layers radial size and interaction length in each layer */
    rdNbOfLayers = (int) pEncoH->scannerTopology.radialNumberOfLayers;

    if ((first_pIntLengthLayers =
	 (double *) calloc(rdNbOfLayers, sizeof(double))) == NULL) {
      printf(ERROR_GEOMETRY8);
      exit(EXIT_FAILURE);
    }

    if ((first_pRdSizeLayers =
	 (double *) calloc(rdNbOfLayers, sizeof(double))) == NULL) {
      printf(ERROR_GEOMETRY8);
      exit(EXIT_FAILURE);
    }

    if (setLayersInfo
	(cch_index, rdNbOfLayers, first_pIntLengthLayers,
	 first_pRdSizeLayers) == 1) {
      printf(ERROR_GEOMETRY1);
      exit(EXIT_FAILURE);
    }

    /* Define an array (called "substructuresNumericalValues") which contains the substructures numerical values: */
    /* layerAxialPitch, crystalAxialPitch, submoduleAxialPitch, moduleAxialPitch, rsectorAxialPitch */
    /* layerRadialPitch, crystalTangentialPitch, submoduleTangentialPitch, moduleTangentialPitch, rsectorTangentialPitch */
    /* axialNumberOfLayers, axialNumberOfCrystals, axialNumberOfSubmodules, axialNumberOfModules, numberOfRings (=axNbOfRsectors) */
    /* radialNbOfLayers, tangentialNbOfCrystals, tangentialNbOfSubmodules, tangentialNbOfModules, numberOfSectors (=tgNbOfRsectors) */

    first_pSubstructuresNumericalValues =
	setSubstructuresValues(pScanGeo, pEncoH);

    /**** Control values of substructurePitches in comparison with structureSizes ****/
    if (testPitchVersusSize
	(first_pSubstructuresNumericalValues, pScanGeo,
	 first_pRdSizeLayers) == 1)
      exit(EXIT_FAILURE);

    /**** Test if(number of rsectors * rsector azimuthal pitch)<= to 360 degrees (or 6.283185307 rad) ****/
    angleDefaultUnitToRadConversionFactor = testAngleDefaultUnit();

    if (((double) (pEncoH->scannerTopology.numberOfSectors - 1) *
	 pScanGeo->rsectorAzimuthalPitch *
	 angleDefaultUnitToRadConversionFactor) > 6.283185307) {
      printf(ERROR_GEOMETRY10, pScanGeo->rsectorAzimuthalPitch,
	     (int) pEncoH->scannerTopology.numberOfSectors);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }

    initializationDone = TRUE;
  }
  /*****************************************************************************************/

  /**** Calculation of the 3D coordinates of substructure (id_t,id_z) in the 3D scanner (x',y',z') system ****/
  /* 1.Calculation of the 3D coordinates of substructure (id_t,id_z) in the tangential reference frame attached to the rsector */
  resultOfCalculOfEventPosition.substructureInRsector3DPosition =
      locateSubstructureInStructure(pcrist, 1,
				    first_pSubstructuresNumericalValues);

  /* 2.Calculation of the 3D coordinates of substructure (id_t,id_z) in the scanner (x',y',z') system */
  rsectorID = locateID(pcrist, 4, first_pSubstructuresNumericalValues);

  rotationAngle =
      (double) (((double) rsectorID.tangential *
		 (pScanGeo->rsectorAzimuthalPitch *
		  angleDefaultUnitToRadConversionFactor)));
  angularRsectorPos =
      (double) ((pScanGeo->azimuthalStep *
		 angleDefaultUnitToRadConversionFactor) *
		(double) pER->gantryAngularPos);

  resultOfCalculOfEventPosition.substructureInScanner3DPosition.radial =
      (-
       (resultOfCalculOfEventPosition.substructureInRsector3DPosition.
	tangential * sin(rotationAngle)));

  resultOfCalculOfEventPosition.substructureInScanner3DPosition.
      tangential =
      (resultOfCalculOfEventPosition.substructureInRsector3DPosition.
       tangential * cos(rotationAngle));

  resultOfCalculOfEventPosition.substructureInScanner3DPosition.axial =
      resultOfCalculOfEventPosition.substructureInRsector3DPosition.axial;

  /**** Calculation of the 3D coordinates of rsector (id_t,id_z) in the 3D laboratory (x,y,z) system ****/
  resultOfCalculOfEventPosition.rsectorInLaboratory3DPosition =
      locateRsectorInLaboratory(pcrist, pScanGeo, pEncoH, pER,
				first_pIntLengthLayers,
				first_pRdSizeLayers,
				angleDefaultUnitToRadConversionFactor,
				first_pSubstructuresNumericalValues);

  /**** Calculation of the 3D coordinates of an event in the 3D laboratory (x,y,z) system ****/
  eventPosX =
      resultOfCalculOfEventPosition.rsectorInLaboratory3DPosition.radial +
      resultOfCalculOfEventPosition.substructureInScanner3DPosition.
      radial + ppShiftValuesList[pcrist[4]][0];

  eventPosY =
      resultOfCalculOfEventPosition.rsectorInLaboratory3DPosition.
      tangential +
      resultOfCalculOfEventPosition.substructureInScanner3DPosition.
      tangential + ppShiftValuesList[pcrist[4]][1];

  eventPosZ =
      resultOfCalculOfEventPosition.rsectorInLaboratory3DPosition.axial +
      resultOfCalculOfEventPosition.substructureInScanner3DPosition.axial +
      ppShiftValuesList[pcrist[4]][2];


  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.radial =
      (eventPosX * cos(angularRsectorPos)) -
      (eventPosY * sin(angularRsectorPos));
  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.tangential =
      (eventPosX * sin(angularRsectorPos)) +
      (eventPosY * cos(angularRsectorPos));
  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.axial =
      eventPosZ;




  free(pcrist);


  return (resultOfCalculOfEventPosition);
}

/**** Destroy the pointers to the lists called "intLengthLayersList" and "rdSizeLayersList" ****/

void destroyScannerGeometryPointers()
{
  if (initializationDone == TRUE) {
    if (first_pIntLengthLayers != NULL)
      free(first_pIntLengthLayers);
    if (first_pRdSizeLayers != NULL)
      free(first_pRdSizeLayers);
    first_pIntLengthLayers = NULL;
    first_pRdSizeLayers = NULL;
    initializationDone = FALSE;
  }
}
