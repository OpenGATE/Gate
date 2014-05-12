/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCalibration_h
#define GateCalibration_h 1

#include "globals.hh"
#include <iostream>
//#include <fstream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateCalibrationMessenger;


/*! \class  GateCalibration
    \brief  Pulse-processor modelling a calibration Nphotons->Energy.

    - GateCalibration - by Martin.Rey@epfl.ch (feb 2003)

    - Can be usefull if you use the class(es) GateLightYield, GateTransferEfficiency or GateQuantumEfficiency.
    You can also set your own calibration factor. Allows the "recalibration" in energy.

      \sa GateVPulseProcessor
*/
class GateCalibration : public GateVPulseProcessor
{
  public:

    //! Constructs a new calibration attached to a GateDigitizer
    GateCalibration(GatePulseProcessorChain* itsChain,
			       const G4String& itsName) ;
    //! Destructor
    virtual ~GateCalibration() ;

  G4double GetCalibrationFactor() { return m_calib; };

  void SetCalibrationFactor(G4double itsCalib) { m_calib = itsCalib; };

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the calibration
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList&  outputPulseList);

  private:
    //! Find the different parameters of the input Pulse :
    //! The depth 'm_depth in the VolumeID 'aVolumeID corresponding @ the volume named 'm_volumeName
    //! The copy number 'm_volumeIDNo of the Inserter of the VolumeID 'aVolumeID for depth 'm_depth
    //! The copy number of the Inserters above: m_k corresponding @ level 'm_depth-1, m_j @ level 'm_depth-2 and m_i @ level 'm_depth-3
    void FindInputPulseParams(const GateVolumeID* aVolumeID);

  private:
    G4String m_volumeName;  //!< Name of the module for quantum efficiency
    G4int m_volumeIDNo;     //!< numero of the volumeID
    G4int m_i, m_j, m_k;    //!< numero of the volumeID
    size_t m_depth;         //!< Depth of the selected volume in the Inserter
    GateCalibrationMessenger *m_messenger;    //!< Messenger
  G4double m_calib;
};


#endif
