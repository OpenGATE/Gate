/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateReadout
    \brief  Pulse-processor modelling a simple PMT Readout (maximum energy wins) of a crystal-block

    - GateReadout - by Daniel.Strul@iphe.unil.ch

    - The Readout is parameterised by its 'depth': pulses will be summed up if their volume IDs
      are identical up to this depth. For instance, the default depth is 1: this means that
      pulses will be considered as taking place in a same block if the first two figures
      of their volume IDs are identical

  S. Stute - June 2014: complete redesign of the Readout module and add a new policy to emulate PMT.
    - Fix bug in choosing the maximum energy pulse.We now have some temporary lists of variables to deal
      with the output pulses. These output pulses are only created at the end of the method. In previous
      versions, the output pulse was accumulating energy as long as input pulses were merged together, but
      the problem is that the comparison of energy was done between the input pulse and this output pulse
      with increasing energy. So with more than 2 pulses to be merged together, the behaviour was undefined.
    - Move all the processing into the upper method ProcessPulseList instead of using the mother version
      working into the ProcessOnePulse method. Thus the ProcessOnePulse in this class is not used anymore.
    - Create policy choice: now we can choose via the messenger between EnergyWinner and EnergyCentroid.
    - For the EnergyCentroid policy, the centroid position is computed using the crystal indices in each
      direction, doing the computation with floating point numbers, and then casting the result into
      integer indices. Using that method, we ensure the centroid position to be in a crystal (if we work
      with global position, we can fall between two crystals in the presence of gaps).
      The depth is ignored with this strategy; it is forced to be at one level above the 'crystal' level.
      If there a 'layer' level below the 'crystal' level, an energy winner strategy is adopted.

  O. Kochebina - April 2022: new messenger options are added and some minor bugs corrected
  O. Kochebina - August 2022: moved to new digitizer
*/


#ifndef GateReadout_h
#define GateReadout_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateReadoutMessenger.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"
#include "GateVSystem.hh"
#include "GateArrayComponent.hh"
#include "GateObjectStore.hh"
#include "GateSinglesDigitizer.hh"



class GateReadout : public GateVDigitizerModule
{
public:

  GateReadout(GateSinglesDigitizer *digitizer, G4String name);
  ~GateReadout();
  
  void Digitize() override;

  //! print-out the attributes specific of the Readout
  void DescribeMyself(size_t );
  //! Returns the depth of the Readout
  inline G4int GetDepth() const  	      	{ return m_depth; }

  //! Set the depth of the Readout
  inline void  SetDepth(G4int aDepth)         { m_depth = aDepth; }

  //! Set the policy of the Readout
  inline void SetPolicy(const G4String& aPolicy)  { m_policy = aPolicy; };
  inline G4String GetPolicy() const  	      	{ return m_policy; }

  //! Set the volume for the Readout
  inline void SetVolumeName(const G4String& aName) { m_volumeName = aName; };
  inline G4String GetVolumeName() const  	      	{ return m_volumeName; }

  //! Set the volume for the Readout even for centroid policy
  inline void ForceDepthCentroid(const G4bool& value) { m_IsForcedDepthCentroid = value; };
  inline G4bool IsDepthForcedCentroid() const  	      	{ return m_IsForcedDepthCentroid; }

  //! Set how the resulting positions should be defined
  inline void SetResultingXY(const G4String& aString) { m_resultingXY= aString;};
  inline G4String GetResultingXY() const  	      	{ return m_resultingXY; };

  inline void SetResultingZ(const G4String& aString){m_resultingZ= aString;};
  inline G4String GetResultingZ() const  	      	{ return m_resultingZ; };

  void SetReadoutParameters();

  //! Reset the local position to be 0
  inline void ResetLocalPos() {m_outputDigi->m_localPos[0]=0.;m_outputDigi->m_localPos[1]=0.;m_outputDigi->m_localPos[2]=0.;}
  void ResetGlobalPos(GateVSystem* system);


private:

  //! The default is the one parameter that defines how a Readout works:
  //! pulses will be summed up if their volume IDs are identical up to this depth.
  //! For instance, the default depth is 1: this means that pulses will be considered as
  //! taking place in a same block if the first two figures of their volume IDs are identical
  G4int m_depth;

  //! S. Stute: add an option to choose the policy of the Readout (using two define integers; see the beginning of this file)
  //G4int m_policy;
  G4String m_policy;
  GateVSystem* m_system;
  G4int m_nbCrystalsX;
  G4int m_nbCrystalsY;
  G4int m_nbCrystalsZ;
  G4int m_nbCrystalsXY;
  G4int m_systemDepth;
  G4int m_crystalDepth;
  GateArrayComponent* m_crystalComponent;

  G4String m_volumeName;
  G4bool m_IsForcedDepthCentroid;

  G4String m_resultingXY;
  G4String m_resultingZ;
  G4bool   m_IsFirstEntrance;//Entrance

  std::vector<int> numberOfComponentForLevel; //!< Table of number of element for each geometric level
  G4int numberOfHigherLevels ;  //!< number of geometric level higher than the one chosen by the user
  G4int numberOfLowerLevels ;  //!< number of geometric level higher than the one chosen by the user
  GateReadoutMessenger *m_messenger;	  //!< Messenger for this

  GateDigi* m_outputDigi;
  GateDigiCollection*  m_OutputDigiCollection;
  GateSinglesDigitizer *m_digitizer;


};

#endif








