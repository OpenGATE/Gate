/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBlurringWithIntrinsicResolution_h
#define GateBlurringWithIntrinsicResolution_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"


class GateBlurringWithIntrinsicResolutionMessenger;

/*! \class  GateBlurringWithIntrinsicResolution
    \brief  Pulse-processor for simulating a special and local Gaussian blurring

    - GateBlurringWithIntrinsicResolution - by Martin.Rey@epfl.ch (mai 2003)

    - Pulse-processor for simulating a Gaussian local blurring on the energy spectrum
    (different for several cystals) based on model :
    \f[R=\sqrt{\frac{1.1}{N_{ph}\cdot QE\cdot TE}\cdot 2.35^2+R_i^2}\f]
    where \f$N_{ph}=LY\cdot E_{inputPulse}\f$.
    You have to give the intrinsic resolution (Ri) of each crystal.
    If you use also the modules for Light Yield, Transfert et Quantum Efficiencies (LY, TE and QE),
    this module takes these differents values, in other case they are equal to 1.

      \sa GateVPulseProcessor
*/
class GateBlurringWithIntrinsicResolution : public GateVPulseProcessor
{

  public:
    //! Public Constructor
    GateBlurringWithIntrinsicResolution(GatePulseProcessorChain* itsChain,
			       const G4String& itsName) ;

    //! Public Destructor
    virtual ~GateBlurringWithIntrinsicResolution() ;

  public:

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.

    G4int ChooseVolume(G4String val);


    //! \name setters and getters
    //@{

    //! Allows to set the intrinsic resolution @ an energy of reference for crystal called 'name'
    void SetIntrinsicResolution(G4String name, G4double val)   { m_table[name].resolution = val; };
    void SetRefEnergy(G4String name, G4double val)   {m_table[name].eref = val; };

    //! Allows to get the intrinsic resolution @ an energy of reference for crystal called 'name'
    //@{
    G4double GetIntrinsicResolution(G4String name)   { return m_table[name].resolution; };
    G4double GetRefEnergy(G4String name)   {return m_table[name].eref; };
    //@}

    //@}

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the blurring
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    //! Find the different parameters of the input Pulse :
    //! The depth 'm_depth in the VolumeID 'aVolumeID corresponding @ the volume named 'm_volumeName
    //! The copy number 'm_volumeIDNo of the Inserter of the VolumeID 'aVolumeID for depth 'm_depth
    //! The copy number of the Inserters above: m_k corresponding @ level 'm_depth-1, m_j @ level 'm_depth-2 and m_i @ level 'm_depth-3
    void FindInputPulseParams(const GateVolumeID* aVolumeID);

    struct param {
      G4double resolution;
      G4double eref;
    };

  private:
    param m_param;                                 //!< Simulated energy resolution and energy of reference
    G4String m_name;                               //!< Name of the volume
    GateMap<G4String,param> m_table ;  //!< Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //!< Table iterator
    G4String m_volumeName;  //!< Name of the module for quantum efficiency
    G4int m_volumeIDNo;     //!< numero of the volumeID
    G4int m_i, m_j, m_k;    //!< numero of the volumeID
    size_t m_depth;         //!< Depth of the selected volume in the Inserter

    GateBlurringWithIntrinsicResolutionMessenger *m_messenger;   //!< Messenger
};


#endif
