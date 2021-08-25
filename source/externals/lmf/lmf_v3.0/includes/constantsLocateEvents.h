/*-------------------------------------------------------

           List Mode Format 
                        
     --  constantsLocateEvents.h  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of constantsLocateEvents.h:
	 Variables and constants used for the calculation of events position.


-------------------------------------------------------*/

/* Definitions of the constants and the structures used to calculate 
   the elements 3D coordinates in the 3D laboratory system (x,y,z) */

#ifndef _constants_structuresGeometry_h
#define _constants_structuresGeometry_h

/*** WARNING: If you modify this structure, you must also modify the functions: 
     fillInStructScannerGeometry and locateSubstructureInStructure ****/

typedef struct {
  /*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  double geometricalDesignType;	/* geometricalDesignType=1 :
				   cylindrical scanner geometry */

  /*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  double ringDiameter;
  double azimuthalStep;		/* attached to the scanner rotation movement */
  double axialStep;		/* attached to the scanner translation movement */

  /*=-=-=-=-=-=-=-=-=-=- RSECTOR -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  double rsectorAxialPitch;
  double rsectorAxialSize;
  double rsectorTangentialSize;
  double rsectorAzimuthalPitch;

  /*=-=-=-=-=-=-=-=-=-=- MODULE =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  double moduleAxialPitch;
  double moduleTangentialPitch;
  double moduleAxialSize;
  double moduleTangentialSize;

  /*=-=-=-=-=-=-=-=-=-=- SUBMODULE  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  double submoduleAxialPitch;
  double submoduleTangentialPitch;
  double submoduleAxialSize;
  double submoduleTangentialSize;

  /*=-=-=-=-=-=-=-=-=-=- CRYSTAL -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  double crystalAxialPitch;
  double crystalTangentialPitch;
  double crystalAxialSize;
  double crystalTangentialSize;
  double crystalRadialSize;
} LMF_cch_scannerGeometry;


typedef struct {
  int tangential;		/* id_substructure modulo tangentialNumberOfSubstructures */
  int axial;			/* id_substructure divided by the tangentialNumberOfSubstructures */
} generalSubstructureID;

typedef struct {
  double radial;		/* the radial position in the tangential reference frame 
				   or the x-coordinate in the 3D coordinates (x,y,z) system */
  double tangential;		/* the tangential position in the tangential reference frame 
				   or the y-coordinate in the 3D coordinates (x,y,z) system */
  double axial;			/* the axial position in the tangential reference frame 
				   or the z-coordinate in the 3D coordinates (x,y,z) system */
} coordinates;

typedef struct {
  coordinates substructureInRsector3DPosition;
  coordinates substructureInScanner3DPosition;
  coordinates rsectorInLaboratory3DPosition;
  coordinates eventInLaboratory3DPosition;
} calculOfEventPosition;

#endif


/* error messages provide by the program used to calculate elements 3D coordinates in the 3D laboratory (x,y,z) system */

#ifndef _error_messages_geometry_h
#define _error_messages_geometry_h

#define ERROR_GEOMETRY1 "ERROR: Initialization of variables used to calculate the 3D position of an event failed!\n"
#define ERROR_GEOMETRY2 "ERROR: This program is available only with cylindric scanner geometry.\n"
#define ERROR_GEOMETRY3 "ERROR: The 2 values: layer ID (=%d) and/or radial number of layers (=%d) are wrong.\nTo continue this program, you must correct the radial number of layers.\n"
#define ERROR_GEOMETRY4 "ERROR: The \"%s\" must be defined to calculate the 3D position of an event.\n"
#define ERROR_GEOMETRY5 "ERROR: The \"%s\" and the \"%s\" must be defined to calculate the 3D position of an event.\n"
#define ERROR_GEOMETRY6 "ERROR: The %s axial pitch and/or the %s axial size are wrong.\n"
#define ERROR_GEOMETRY7 "ERROR: The %s tangential pitch and/or the %s tangential size are wrong.\n"
#define ERROR_GEOMETRY8 "ERROR: In the function \"locateEventPosition\", allocation of memory failed !!"
#define ERROR_GEOMETRY9 "ERROR: In the function \"fillInLayersInfoList\", allocation of memory failed !!"
#define ERROR_GEOMETRY10 "ERROR: The rsector azimuthal pitch (=%f) and/or the number of sectors per ring (=%c) are wrong.\n"
#define ERROR_GEOMETRY11 "ERROR: The crystal radial size and/or the layers radial size are wrong.\n"

#endif
