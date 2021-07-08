/*-------------------------------------------------------

           List Mode Format 
                        
     --  prototypesLocateEvents.h  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of prototypesLocateEvents.h:

	 Functions used for the calculation of events position 
	 in the 3D laboratory (x.y.z) system:
	 ->locateEventInLaboratory - Compute the elements 3D coordinates 
	                             in the laboratory (x,y,z) system
	 ->fillInStructScannerGeometry - Initialize the members of the structure 
	                                 LMF_cch_scannerGeometry 
	 ->setLayersInfo - Initialize the radial size and the interaction length in each layer
	 ->setSubstructuresValues - Create an array ("substructureNumericalValues"),
	   which contains the substructures numerical values
	 ->testPitchVersusSize - Control the value of the substructures pitches in comparison 
	                         with the substructures sizes
	 ->testAngleDefaultUnit - Test if the angle default unit is the radian
	 ->locateID - Define the crystal or the submodule or the module or the rsector tangential and axial ID
	 ->locateSubstructureInStructure - Compute the 3D coordinates of substructure(id_t,id_z) 
	                                   in a tangential reference frame attached to a rsector
	 ->locateRsectorInLaboratory - Process to calculate the 3D coordinates of a rsector in the laboratory (x,y,z) system
	 ->destroyScannerGeometryPointers - Destroy the pointers to the lists called "intLengthLayersList"
	                                    and "rdSizeLayersList" 

---------------------------------------------------------------------------*/

/* locateEventInLaboratory - Calculation of the elements 3D coordinates in the 3D laboratory (x,y,z) system */

#ifndef _locateEventInLaboratory_h
#define _locateEventInlaboratory_h

calculOfEventPosition locateEventInLaboratory(const ENCODING_HEADER *,
					      const EVENT_RECORD *, int);

#endif


/* fillInStructScannerGeometry - Initialize the members of the structure LMF_cch_scannerGeometry */

#ifndef _fillInStructScannerGeometry_h
#define _fillInStructScannerGeometry_h

int fillInStructScannerGeometry(int cch_index,
				LMF_cch_scannerGeometry * pScanGeo);

#endif

/* setLayersInfo - Initialize the radial size and the interaction length in each layer */

#ifndef _setLayersInfo_h
#define _setLayersInfo_h

int setLayersInfo(int cch_index,
		  int rdNbOfLayers,
		  double *first_pIntLengthLayers,
		  double *first_pRdSizeLayers);

#endif

/* setSubstructuresValues - Create an array ("substructureNumericalValues"), 
   which contains the substructures numerical values: */
/* layerAxialPitch, crystalAxialPitch, submoduleAxialPitch, moduleAxialPitch, rsectorAxialPitch */
/* layerRadialPitch, crystalTangentialPitch, submoduleTangentialPitch, moduleTangentialPitch, rsectorTangentialPitch */
/* axialNumberOfLayers, axialNumberOfCrystals, axialNumberOfSubmodules, axialNumberOfModules, axialNumberOfRsectors */
/* radialNbOfLayers, tangentialNbOfCrystals, tangentialNbOfSubmodules, tangentialNbOfModules, tangentialNbOfRsectors */

#ifndef _setSubstructuresValues_h
#define _setSubstructuresValues_h

double *setSubstructuresValues(LMF_cch_scannerGeometry *,
			       const ENCODING_HEADER *);

#endif

/* testPitchVersusSize - Control the value of the substructures pitches in comparison with the substructures sizes */

#ifndef _testPitchVersusSize_h
#define _testPitchVersusSize_h

int testPitchVersusSize(double *first_pSubstructuresNumericalValues,
			LMF_cch_scannerGeometry * pScanGeo,
			double *first_pRdSizeOfLayers);

#endif

/* testAngleDefaultUnit - Test if the angle default unit is the radian */

#ifndef _testAngleDefaultUnit_h
#define _testAngleDefaultUnit_h

double testAngleDefaultUnit();

#endif

/* locateID - Define the crystal or the submodule or the module or the rsector tangential and axial ID */

#ifndef _locateID_h
#define _locateID_h

generalSubstructureID locateID(u16 * pcrist, int substructureOrder, double
			       *first_pSubstructuresNumericalValues);

u16 remakeID(generalSubstructureID substructureID,
	     int substructureOrder,
	     double *first_pSubstructuresNumericalValues);

#endif

/* locateSubstructureInStructure - Calculation of the 3D coordinates of substructure(id_t,id_z) 
   in a tangential reference frame attached to a rsector */

#ifndef _locateSubstructureInStructure_h
#define _locateSubstructureInStructure_h


coordinates locateSubstructureInStructure(u16 * pcrist,
					  int substructureOrder, double
					  *first_pSubstructuresNumericalValues);

#endif

/* locateRsectorInLaboratory - Process to calculate the 3D coordinates of a rsector 
   in the 3D laboratory (x,y,z) system */

#ifndef _locateRsectorInLaboratory_h
#define _locateRsectorInLaboratory_h

coordinates locateRsectorInLaboratory(u16 *,
				      LMF_cch_scannerGeometry *,
				      const ENCODING_HEADER *,
				      const EVENT_RECORD *,
				      double *, double *,
				      double, double *);

#endif



/* destroyScannerGeometryPointers - Destroy the pointers to the lists called "intLengthLayersList" 
   and "rdSizeLayersList" */

#ifndef _destroyScannerGeometryPointers_h
#define _destroyScannerGeometryPointers_h

void destroyScannerGeometryPointers();

#endif
