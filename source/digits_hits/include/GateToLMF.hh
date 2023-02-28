/*----------------------
   OpenGATE Collaboration

   Luc Simon <luc.simon@iphe.unil.ch>
   Daniel Strul <daniel.strul@iphe.unil.ch>
   Giovanni Santin <giovanni.santin@cern.ch>
   Claude Comtat <comtat@ieee.org>
   Jean-Marc Vieira <jean-marc.vieira@epfl.ch>

   Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne
   Copyright (C) 2003 CEA/SHFJ, F-91401 Orsay
   Copyright (C) 2003 EPFL/LPHE, CH-1015 Lausanne

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GateToLMF_h
#define GateToLMF_h 1

#include "GateConfiguration.h"

#ifdef GATE_USE_LMF

#include <iostream>
#include <stdio.h>
#include "GateVOutputModule.hh"
#include "G4UserEventAction.hh"
#include "GateDigi.hh"
#include "globals.hh"
#include "lmf_gate.h"

class GateVSystem;
class GateToLMFMessenger;
class GateCoincidenceDigi;

class GateToLMF : public GateVOutputModule
{
public :

  GateToLMF(const G4String& name, GateOutputMgr* outputMgr,GateVSystem *pSystem,DigiMode digiMode); //!< Constructor
  virtual ~GateToLMF(); //!< Destructor

public :


  void RecordBeginOfAcquisition();    	       //!< Function called at the beginning of a new acquisition
  void RecordEndOfAcquisition();      	       //!< Function called at the end of an acquisition
  void RecordBeginOfRun(const G4Run *);        //!< This function writes the ASCII header
  void RecordEndOfRun(const G4Run *) {};       //!< This function doesn't do anything.
  void RecordBeginOfEvent(const G4Event *) {}; //!< This function doesn't do anything.
  void RecordEndOfEvent(const G4Event *);      //!< This function gives the digis to LMF.
  void RecordStep(const G4Step *) {};          //!< This function doesn't do anything.
	const G4String& GiveNameOfFile(){ return m_nameOfFile; };          //!< This function doesn't do anything.
	void RecordStepWithVolume(const GateVVolume *, const G4Step *) {}; //!< This function doesn't do anything.

  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *) {};

  //! To return energy value.
  u8 GetEnergy(u8 id){ return m_LMFEnergy[id]; };
  //! To convert energy value in LMF units (keV).
  /*! Exemple : if energy = 0.438256 MeV
    and the LMF enegy step is 2 kev, the energy becomes 22.*/
  void SetEnergy(u8 id, G4double value) {
    m_LMFEnergy[id] = (u8)(((value/GATE_LMF_ENERGY_STEP_KEV)/keV)+0.5); };

  //! To return time array pointer (not a big interest !!!).
  u8 *GetTime(u8 id) {return (m_pLMFTime[id]); };
  //!  To convert time value in LMF units (picosecond).
  /*! Exemple : if time = 2.4438256 nanosecond
    and the LMF time step is 2 picosecond, the time becomes 1222.  */
  void SetTime(u8 id, G4double value);

  u16 GetGantryAxialPos();   //!< returns gantry's axial position
  u16 GetGantryAngularPos(); //!< returns gantry's angular position
  u16 GetSourceAxialPos(); //!< returns source's axial position
  u16 GetSourceAngularPos(); //!< returns source's angular position

  void SetGantryAxialPos(G4int value);//!< set gantry's axial position
  void SetGantryAngularPos(G4int value);//!< set gantry's angular position
  void SetSourceAxialPos(G4int value);//!< set source's axial position
  void SetSourceAngularPos(G4int value);//!< set source's angular position

  u16 GetLayerID(u8 id) { return m_LMFLayerID[id]; };         //!< To return ID values.
  u16 GetCrystalID(u8 id) { return m_LMFCrystalID[id]; };     //!< To return ID values.
  u16 GetSubmoduleID(u8 id) { return m_LMFSubmoduleID[id]; }; //!< To return ID values.
  u16 GetModuleID(u8 id) { return m_LMFModuleID[id]; };       //!< To return ID values.
  u16 GetRsectorID(u8 id) { return m_LMFRsectorID[id]; };     //!< To return ID values.


  void SetLayerID(u8 id, G4int value) { m_LMFLayerID[id] = (u16) value; };         //!< To set ID values.
  void SetCrystalID(u8 id, G4int value) { m_LMFCrystalID[id] = (u16) value; };     //!< To set ID values.
  void SetSubmoduleID(u8 id, G4int value) { m_LMFSubmoduleID[id] = (u16) value; }; //!< To set ID values.
  void SetModuleID(u8 id, G4int value) { m_LMFModuleID[id] = (u16) value; };       //!< To set ID values.
  void SetRsectorID(u8 id, G4int value) { m_LMFRsectorID[id] = (u16) value; };     //!< To set ID values.


  //! To set the output file name (used by GateToLMFMessenger)
  void SetOutputFileName(G4String ofname);

  //! Get the input data channel name
  const  G4String& GetInputDataName()
    { return m_inputDataChannel;       };
  //! Set the output data channel name
  void   SetOutputDataName(const G4String& aName)
    { m_inputDataChannel = aName;      };


  /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetCoincidenceBool(G4bool value);
   /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetDetectorIDBool(G4bool value);
   /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetEnergyBool(G4bool value);
   /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetNeighbourBool(G4bool value);
  /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */

  void SetNeighbourhoodOrder(G4int value);
   /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
   void SetGantryAxialPosBool(G4bool value);
   /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetGantryAngularPosBool(G4bool value);
   /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetSourcePosBool(G4bool value);
   /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetGateDigiBool(G4bool value);
    /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetComptonBool(G4bool value);
  /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetComptonDetectorBool(G4bool value);
  /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetSourceIDBool(G4bool value);
  /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetSourceXYZPosBool(G4bool value);
  /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetGlobalXYZPosBool(G4bool value);
  /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetEventIDBool(G4bool value);
    /*!
    If the auto header is not use you can configure the LMF headers with these
    messenger function.
    \sa GateToLMFMessenger
   */
  void SetRunIDBool(G4bool value);

  //! Display a digi in LMF units.
  void showOneLMFDigi();
  //! Convert the digi in a LMF units, calling libLMF.a.
  void buildLMFEventRecord();


  /*!
    This function creates the ASCII LMF file
    from the psystem pointer.
  */
  void createLMF_ASCIIfile(void);


public :
  //! This function takes a single digi and calls the different functions of this class.
  /*!
    It transforms the digi to be compatible with the LMF C-library.
  \param digi is a single digi
  \sa GateDigi
  \sa GateToDigi
  \sa SetEnergy()
  \sa SetCrystalID()
  \sa SetModuleID()
  \return nothing.
  */
  void StoreTheDigiInLMF(GateDigi *digi);
  //! This function fills the extra infos that can give GATE in the LMF_ccs_gateDigiRecord
  void StoreMoreDigiInLMF_GDR(GateDigi *digi);

  void StoreTheCoinciDigiInLMF(GateCoincidenceDigi *digi);

private :
  //! Energy in LMF unit.
  u8 m_LMFEnergy[2];
  //! Time array...
  /*!
    This "8 characters array" can contain the integer value of time in picosecond
    We use this not very useful way to store time in LMF for differents reasons.
    Mainly, it's more easy to write one only byte in a binary file.
   */
  u8 m_pLMFTime[8][2];


  u16 m_LMFLayerID[2]; //!< Layer ID in a crystal
  u16 m_LMFCrystalID[2]; //!< Crystal ID in a submodule
  u16 m_LMFSubmoduleID[2]; //!< Submodule ID in a Module
  u16 m_LMFModuleID[2];  //!< Module ID in a rSector
  u16 m_LMFRsectorID[2];  //!< rSector ID



  u16 m_LMFgantryAxialPos;  //!< gantry's axial position, 16 bits
  u16 m_LMFgantryAngularPos; //!< gantry's angular position, 16 bits
  u16 m_LMFsourceAngularPos;//!< external source's angular position, 16 bits
  u16 m_LMFsourceAxialPos;  //!< external source's axial position, 16 bits



  G4double m_azimuthalStep;	      	//!< Azimuthal step (in Geant4 angular units)
  G4double m_axialStep;	      	      	//!< Aximuthal step (in Geant4 axial units)

  // eccentric rotation parameters
  G4double m_shiftX;	            	//!< Shift along X
  G4double m_shiftY;	            	//!< Shift along Y
  G4double m_shiftZ;	            	//!< Shift along Z (by default not used, added in case off ...)

  G4double m_pZshift_vector[256] ;
  G4int    m_RingModuloNumber;


  ENCODING_HEADER *pEncoH ; //----------------
  EVENT_HEADER *pEH ;       // Declaration of|
  GATE_DIGI_HEADER *pGDH ;   //               |
  COUNT_RATE_HEADER *pCRH ;  //  a LMF Record |
  CURRENT_CONTENT *pcC ;    //    Carrier    |
  EVENT_RECORD *pER[2] ;       //               |
  EVENT_RECORD *pERC ;       //               |
  COUNT_RATE_RECORD *pCRR ;  //----------------


  FILE *m_pfile,*m_pASCIIfile;
  G4String m_nameOfFile,m_nameOfASCIIfile,m_name;
  /*!
    The messenger is for scripted UI.
    \sa GateOutputModuleMessenger
    \sa GateToLMFMessenger
    \sa GateMessenger
    \sa G4Messenger
   */
  GateToLMFMessenger *m_LMFMessenger;


  //! pointer to system
  /*!
    To get the scanner topology, we need a pointer to the system.
    \sa GateVSystem
    \sa GateCylindricalPETSystem
  */
  GateVSystem *m_pSystem;

  G4String m_inputDataChannel;	  //!< Name of the coincidence-collection to store into the LMF files
  G4int    m_inputDataChannelID;	  //!< ID of the coincidence-collection to store into the LMF files

};


#endif
#endif
