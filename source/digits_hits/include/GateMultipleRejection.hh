/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  The purpose of this class is to help to create new users digitizer module(DM).

  \class  GateMultipleRejection

  Last modification (Adaptation to GND): August 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateMultipleRejection_h
#define GateMultipleRejection_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateMultipleRejectionMessenger.hh"
#include "GateSinglesDigitizer.hh"

typedef enum {kvolumeName,
              kvolumeID,
             } multiple_def_t;


class GateMultipleRejection : public GateVDigitizerModule
{
public:
  
  GateMultipleRejection(GateSinglesDigitizer *digitizer, G4String name);
  ~GateMultipleRejection();
  
  void Digitize() override;

  void SetMultipleRejection(G4bool val)   { m_MultipleRejection = val; };

  void DescribeMyself(size_t );


  void SetMultipleDefinition( G4String policy){
      if (policy=="volumeID"){
          m_multipleDef = kvolumeID;
      }
      else {
          if(policy=="volumeName"){
              m_multipleDef = kvolumeName;
          }
          else{
              G4cout<<"WARNING : multiple rejection policy not recognized. Default volumeName policy is employed \n";
          }
      }
  }


protected:

  //G4bool m_rejectionAllPolicy;
  multiple_def_t m_multipleDef;

  std::vector<int> m_multiplesIndex;
  G4bool m_multiplesRejPol;

  std::vector< GateDigi* >* m_waiting;

  G4bool  m_MultipleRejection;

private:
  GateDigi* m_outputDigi;

  GateMultipleRejectionMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;

  std::vector<G4String> m_VolumeNames;
  std::vector<GateVolumeID> m_VolumeIDs;


 // int currentNumber;
 // G4String currentVolumeName;
 // G4String m_name;
};

#endif








