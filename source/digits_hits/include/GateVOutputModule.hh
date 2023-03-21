/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateVOutputModule_H
#define GateVOutputModule_H


#include "GateConfiguration.h"

#include "globals.hh"

class G4Run;
class G4Step;
class G4Event;

class GateOutputMgr;
class GateVGeometryVoxelStore;
class GateSteppingAction;
class GateVVolume;

enum DigiMode {
  kruntimeMode,
  kofflineMode
};

class GateVOutputModule
{
public:

  GateVOutputModule(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode);

  virtual ~GateVOutputModule();

  virtual void RecordBeginOfAcquisition() = 0;
  virtual void RecordEndOfAcquisition() = 0;
  virtual void RecordBeginOfRun(const G4Run *) = 0;
  virtual void RecordEndOfRun(const G4Run *) = 0;
  virtual void RecordBeginOfEvent(const G4Event *) = 0;
  virtual void RecordEndOfEvent(const G4Event *) = 0;
  virtual void RecordStepWithVolume(const GateVVolume * v, const G4Step *) = 0;
  virtual void RecordVoxels(GateVGeometryVoxelStore *) = 0;

  virtual void RecordTracks(GateSteppingAction*){} /* PY Descourt 08/09/2009 */

  virtual void SetVerboseLevel(G4int val) { nVerboseLevel = val; }

/*
 * Pure virtual method that return the name of the output fileName.
 * By default all output name for each module that has an output
 * file is " ". So this method will be call in startDAQ for each
 * OutputModule to check if the filenames are given or not.
*/
  virtual const G4String& GiveNameOfFile() = 0;

  /*! \brief Virtual method to print-out a description of the module

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void Describe(size_t indent=0);

  inline G4String GetName()              { return m_name; }
  inline void     SetName(G4String name) { m_name = name; }

  inline GateOutputMgr* GetOutputMgr() { return m_outputMgr; }

  inline DigiMode GetDigiMode() const 	  { return m_digiMode;}
  inline void SetDigiMode(DigiMode mode)  { m_digiMode = mode; }

  /*virtual void RegisterNewCoincidenceDigiCollection(const G4String& aCollectionName,G4bool outputFlag)
    {;}
    virtual void RegisterNewSingleDigiCollection(const G4String& aCollectionName,G4bool outputFlag)
    {;}*/
  virtual void RegisterNewCoincidenceDigiCollection(const G4String& ,G4bool )
  {}
  virtual void RegisterNewSingleDigiCollection(const G4String& ,G4bool )
  {}
  //OK GND 2022
   virtual void RegisterNewHitsCollection(const G4String& ,G4bool )
   {}
  //! Returns the value of the object enabled/disabled status flag
  inline virtual G4bool IsEnabled() const
  { return m_isEnabled;}
  //! Enable the object
  inline virtual void Enable(G4bool val)
  { m_isEnabled = val; }

  //OK GND 2022
    virtual G4int GetCollectionID(G4String);
protected:

  G4int                      nVerboseLevel;

  GateOutputMgr*             m_outputMgr;

  G4String                   m_name;

  DigiMode    	      	     m_digiMode;

  //! Flag telling whether the object is enabled (active) or disabled (off)
  G4bool 		     m_isEnabled;
};
//---------------------------------------------------------------------------

#endif
